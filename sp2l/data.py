"""Data loading and synthetic data generation."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .models import Bar

_TS_KEYS = ("time", "timestamp", "datetime", "date", "ts")


def _parse_ts(raw: str) -> datetime:
    raw = raw.strip()
    if raw.replace(".", "", 1).isdigit():
        val = float(raw)
        if val > 1e12:  # milliseconds
            val /= 1000.0
        return datetime.fromtimestamp(val, tz=timezone.utc)
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y.%m.%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.fromisoformat(raw)


def load_csv(path: str) -> List[Bar]:
    """Load OHLCV bars from a CSV with headers (case-insensitive):
    time/timestamp/date + open, high, low, close [, volume]."""
    bars: List[Bar] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        fields = {name.lower().strip(): name for name in reader.fieldnames}
        ts_key = next((fields[k] for k in _TS_KEYS if k in fields), None)
        if ts_key is None:
            raise ValueError(f"{path}: no timestamp column found ({reader.fieldnames})")
        for col in ("open", "high", "low", "close"):
            if col not in fields:
                raise ValueError(f"{path}: missing column '{col}'")
        vol_key = fields.get("volume") or fields.get("vol")
        for row in reader:
            bars.append(
                Bar(
                    ts=_parse_ts(row[ts_key]),
                    open=float(row[fields["open"]]),
                    high=float(row[fields["high"]]),
                    low=float(row[fields["low"]]),
                    close=float(row[fields["close"]]),
                    volume=float(row[vol_key]) if vol_key and row.get(vol_key) else 0.0,
                )
            )
    bars.sort(key=lambda b: b.ts)
    return bars


def synthetic_bars(
    n: int = 2000,
    seed: int = 42,
    start_price: float = 2400.0,
    start_ts: Optional[datetime] = None,
    bar_minutes: int = 5,
    episode_every: int = 40,
) -> List[Bar]:
    """Random-walk M5-style series with injected SP2L-like episodes
    (spike -> pullback -> second leg) so demos/backtests have setups to find.
    Roughly calibrated to gold (XAUUSD) volatility."""
    rng = random.Random(seed)
    ts = start_ts or datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    price = start_price
    base_vol = start_price * 0.0004
    bars: List[Bar] = []
    i = 0
    while i < n:
        inject = i > 20 and i % episode_every == 0
        if inject:
            direction = 1 if rng.random() < 0.5 else -1
            follow_through = rng.random() < 0.6  # some episodes fail on purpose
            # Spike: 3-5 strong candles with an FVG in the middle.
            spike_bars = rng.randint(3, 5)
            for k in range(spike_bars):
                body = base_vol * rng.uniform(2.0, 3.5)
                gap = base_vol * rng.uniform(0.6, 1.2) if k == spike_bars // 2 else 0.0
                o = price + direction * gap
                c = o + direction * body
                hi = max(o, c) + base_vol * rng.uniform(0.0, 0.3)
                lo = min(o, c) - base_vol * rng.uniform(0.0, 0.3)
                bars.append(Bar(ts, o, hi, lo, c, rng.uniform(500, 1500)))
                ts += timedelta(minutes=bar_minutes)
                price = c
                i += 1
            # Pullback: 2-4 counter candles retracing ~30-60%.
            pull_bars = rng.randint(2, 4)
            for _ in range(pull_bars):
                body = base_vol * rng.uniform(0.8, 1.6)
                o = price
                c = o - direction * body
                hi = max(o, c) + base_vol * rng.uniform(0.0, 0.4)
                lo = min(o, c) - base_vol * rng.uniform(0.0, 0.4)
                bars.append(Bar(ts, o, hi, lo, c, rng.uniform(300, 800)))
                ts += timedelta(minutes=bar_minutes)
                price = c
                i += 1
            # Second leg (or failure).
            leg_bars = rng.randint(3, 6)
            leg_dir = direction if follow_through else -direction
            for _ in range(leg_bars):
                body = base_vol * rng.uniform(1.2, 2.5)
                o = price
                c = o + leg_dir * body
                hi = max(o, c) + base_vol * rng.uniform(0.0, 0.4)
                lo = min(o, c) - base_vol * rng.uniform(0.0, 0.4)
                bars.append(Bar(ts, o, hi, lo, c, rng.uniform(400, 1200)))
                ts += timedelta(minutes=bar_minutes)
                price = c
                i += 1
        else:
            drift = base_vol * rng.uniform(-0.8, 0.8)
            o = price
            c = o + drift
            hi = max(o, c) + base_vol * rng.uniform(0.1, 0.6)
            lo = min(o, c) - base_vol * rng.uniform(0.1, 0.6)
            bars.append(Bar(ts, o, hi, lo, c, rng.uniform(200, 600)))
            ts += timedelta(minutes=bar_minutes)
            price = c
            i += 1
    return bars[:n]


def save_csv(bars: List[Bar], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow(
                [b.ts.strftime("%Y-%m-%d %H:%M:%S"), b.open, b.high, b.low, b.close, b.volume]
            )
