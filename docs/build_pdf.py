#!/usr/bin/env python3
"""Render the Persian (RTL) HTML guides in this folder to PDF.

Requires the Vazirmatn font to be installed system-wide and a Chromium
binary. Usage:  python3 docs/build_pdf.py
"""

import pathlib
import sys

from playwright.sync_api import sync_playwright

# Chromium shipped with the environment; override with the first CLI argument.
CHROME = sys.argv[1] if len(sys.argv) > 1 else "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
HERE = pathlib.Path(__file__).parent
GUIDES = ["guide_tradingview", "guide_mt5"]


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 794, "height": 1123})
        for name in GUIDES:
            src = HERE / f"{name}.html"
            page.goto(src.resolve().as_uri(), wait_until="networkidle")
            # Guard against a section spilling onto an extra page.
            tallest = page.evaluate(
                "() => Math.max(...Array.from(document.querySelectorAll('.page'))"
                ".map(e => e.scrollHeight))"
            )
            if tallest > 1123:
                print(f"warning: {name} has a section {tallest - 1123}px too tall")
            out = HERE / f"{name}.pdf"
            page.pdf(path=str(out), format="A4", print_background=True,
                     display_header_footer=False,
                     margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"})
            print(f"wrote {out.name}")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
