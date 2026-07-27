//+------------------------------------------------------------------+
//|                                        SP2L_Poursamadi_EA.mq5     |
//|  SP2L (Spike - 2 Leg - Level) expert advisor after Mohammad Ali   |
//|  Poursamadi. Mirrors the Python engine in this repository         |
//|  (sp2l/strategy.py + sp2l/backtest.py) and the TradingView        |
//|  indicator (pine/sp2l_indicator.pine).                            |
//|                                                                   |
//|  Logic: spike (consecutive strong candles + P-Gap/FVG + ATR       |
//|  power) -> pullback -> break of level B -> entry, with the stop   |
//|  behind the spike origin (A) and a configurable R:R target.       |
//|                                                                   |
//|  Four entry triggers (BreakConfirm x EntryOrder):                 |
//|    1. Shadow or close + Market -> fills at the next bar's open    |
//|    2. Shadow or close + Limit  -> limit resting at B              |
//|    3. Close only      + Market -> fills at the next bar's open    |
//|    4. Close only      + Limit  -> limit resting at B              |
//|                                                                   |
//|  EDUCATIONAL / RESEARCH USE. Test on a demo account first.        |
//+------------------------------------------------------------------+
#property copyright "SP2L research implementation"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>

//--- inputs ---------------------------------------------------------
input group "Spike detection"
input int    MinSpikeBars      = 3;      // Minimum spike bars
input double MinBodyRatio      = 0.5;    // Min body/range ratio
input int    DojiTolerance     = 1;      // Doji tolerance inside the spike
input double MaxDojiBodyRatio  = 0.25;   // Max body/range that counts as doji
input bool   RequireGap        = true;   // Require a P-Gap (FVG) in the spike
input double PowerMult         = 1.5;    // Spike length >= PowerMult * ATR (0 = off)
input int    AtrLen            = 14;     // ATR length

input group "Setup lifecycle"
input double MaxRetrace        = 1.0;    // Max pullback retrace (x spike)
input int    MaxWaitBars       = 20;     // Max bars to wait after the spike
input double RiskReward        = 1.0;    // Final target (R:R)
input double SlBufferPct       = 0.0;    // SL buffer (fraction of the spike)

input group "Entry trigger"
enum ENUM_BREAK_CONFIRM { BREAK_CLOSE = 0, BREAK_TOUCH = 1 };
enum ENUM_ENTRY_ORDER   { ENTRY_MARKET = 0, ENTRY_LIMIT = 1 };
input ENUM_BREAK_CONFIRM BreakConfirm = BREAK_CLOSE;   // Break of B counts on
input ENUM_ENTRY_ORDER   EntryOrder   = ENTRY_MARKET;  // Entry order type
input int    LimitWaitBars     = 10;     // Limit: bars to wait for a fill
input bool   AllowConcurrent   = false;  // Take all trades (ignore open positions)

input group "Risk & management"
input double RiskPercent       = 1.0;    // Risk per trade (% of equity)
input bool   UsePartial        = false;  // Partial take-profit
input double PartialRR         = 1.0;    // Partial level (R)
input double PartialPercent    = 50.0;   // Partial size (% of the position)
enum ENUM_BE_MODE { BE_OFF = 0, BE_RR = 1, BE_AFTER_PARTIAL = 2 };
input ENUM_BE_MODE BeMode      = BE_OFF; // Breakeven (risk-free) mode
input double BeTriggerRR       = 1.0;    // BE trigger (R)
input double BeOffsetR         = 0.0;    // BE offset (R locked in)
input double TrailAtrMult      = 0.0;    // ATR trailing stop (0 = off)

input group "Guards & session"
input int    MaxTradesPerDay   = 0;      // Max trades per day (0 = off)
input double MaxDailyLossR     = 0.0;    // Max daily loss in R (0 = off)
input bool   UseSession        = false;  // Session filter
input string SessionStart      = "13:30";// Session start (server time)
input string SessionEnd        = "20:00";// Session end (server time)

input group "Execution"
input ulong  MagicNumber       = 20260727;
input int    SlippagePoints    = 20;
input bool   ShowPanel         = true;   // On-chart status panel

//--- globals --------------------------------------------------------
CTrade         trade;
CPositionInfo  posInfo;
COrderInfo     ordInfo;

int      atrHandle = INVALID_HANDLE;
datetime lastBarTime = 0;

// Setup state machine (mirrors the Pine/Python state).
int      g_dir       = 0;      // 0 idle, 1 long setup, -1 short setup
double   g_pointA    = 0.0;    // spike origin (stop anchor)
double   g_pointB    = 0.0;    // spike extreme (entry level)
int      g_spikeEndBars = 0;   // bars since the spike end (shift of that bar)
bool     g_inPull    = false;
int      g_pullBars  = 0;      // bars since the pullback started

// Per-position management memory.
struct PosState
{
   ulong  ticket;
   double entry;
   double initSL;
   bool   partialDone;
   bool   beDone;
};
PosState g_pos[];

// Daily counters.
datetime g_dayStart      = 0;
int      g_dayTrades     = 0;
double   g_dayRiskMoney  = 0.0;  // risk money used as the R unit for the day

//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(SlippagePoints);
   trade.SetTypeFillingBySymbol(_Symbol);

   atrHandle = iATR(_Symbol, PERIOD_CURRENT, AtrLen);
   if(atrHandle == INVALID_HANDLE)
   {
      Print("SP2L: failed to create the ATR handle");
      return(INIT_FAILED);
   }
   if(MinSpikeBars < 2)
   {
      Print("SP2L: MinSpikeBars must be >= 2");
      return(INIT_PARAMETERS_INCORRECT);
   }
   if(RiskPercent <= 0.0)
   {
      Print("SP2L: RiskPercent must be positive");
      return(INIT_PARAMETERS_INCORRECT);
   }
   ResetDayIfNeeded();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(atrHandle != INVALID_HANDLE)
      IndicatorRelease(atrHandle);
   Comment("");
}

//+------------------------------------------------------------------+
//| Main loop: manage positions every tick, run the setup logic once |
//| per closed bar (all decisions use completed candles).            |
//+------------------------------------------------------------------+
void OnTick()
{
   ResetDayIfNeeded();
   SyncPositionStates();
   ManageOpenPositions();

   datetime t = (datetime)SeriesInfoInteger(_Symbol, PERIOD_CURRENT, SERIES_LASTBAR_DATE);
   if(t == lastBarTime)
   {
      if(ShowPanel) DrawPanel();
      return;
   }
   lastBarTime = t;

   OnNewBar();
   if(ShowPanel) DrawPanel();
}

//+------------------------------------------------------------------+
//| One evaluation per newly opened bar. Shift 1 is the bar that just  |
//| closed, shift 0 is the bar opening right now - so a market entry   |
//| placed here IS the "next bar's open" relative to the break bar.    |
//+------------------------------------------------------------------+
void OnNewBar()
{
   ExpireStaleLimits();

   // 1) setup invalidation / expiry
   if(g_dir != 0)
   {
      g_spikeEndBars++;
      if(g_inPull) g_pullBars++;

      double spLen  = MathAbs(g_pointB - g_pointA);
      double hi     = iHigh(_Symbol, PERIOD_CURRENT, 1);
      double lo     = iLow(_Symbol, PERIOD_CURRENT, 1);
      bool   invalid = (g_dir == 1)
                       ? (lo <= g_pointB - spLen * MaxRetrace)
                       : (hi >= g_pointB + spLen * MaxRetrace);
      if(invalid || g_spikeEndBars > MaxWaitBars)
         ResetSetup();
   }

   // 2) spike extension (a contiguous strong candle before any pullback)
   if(g_dir != 0 && !g_inPull && g_spikeEndBars == 1)
   {
      if(IsStrong(1, g_dir))
      {
         g_spikeEndBars = 0;
         g_pointB = (g_dir == 1) ? iHigh(_Symbol, PERIOD_CURRENT, 1)
                                 : iLow(_Symbol, PERIOD_CURRENT, 1);
      }
   }

   // 3) pullback start
   if(g_dir != 0 && !g_inPull && g_spikeEndBars >= 1)
   {
      double o = iOpen(_Symbol, PERIOD_CURRENT, 1);
      double c = iClose(_Symbol, PERIOD_CURRENT, 1);
      double lo = iLow(_Symbol, PERIOD_CURRENT, 1);
      double hi = iHigh(_Symbol, PERIOD_CURRENT, 1);
      double loPrev = iLow(_Symbol, PERIOD_CURRENT, 2);
      double hiPrev = iHigh(_Symbol, PERIOD_CURRENT, 2);
      bool pb = (g_dir == 1) ? (lo < loPrev || c < o) : (hi > hiPrev || c > o);
      if(pb)
      {
         g_inPull   = true;
         g_pullBars = 0;
      }
   }

   // 4) break of B -> place the entry
   if(g_dir != 0 && g_inPull && g_pullBars >= 1)
   {
      double c  = iClose(_Symbol, PERIOD_CURRENT, 1);
      double hi = iHigh(_Symbol, PERIOD_CURRENT, 1);
      double lo = iLow(_Symbol, PERIOD_CURRENT, 1);
      bool broke;
      if(g_dir == 1)
         broke = (BreakConfirm == BREAK_CLOSE) ? (c > g_pointB) : (hi > g_pointB);
      else
         broke = (BreakConfirm == BREAK_CLOSE) ? (c < g_pointB) : (lo < g_pointB);

      if(broke)
      {
         int    d      = g_dir;
         double spLen  = MathAbs(g_pointB - g_pointA);
         double buffer = spLen * SlBufferPct;
         double stop   = (d == 1) ? g_pointA - buffer : g_pointA + buffer;
         double levelB = g_pointB;
         ResetSetup();   // the break consumes the setup either way

         if(!InSession())
            Print("SP2L: signal skipped (outside the session)");
         else if(Busy() && !AllowConcurrent)
            Print("SP2L: signal skipped (already in a trade / order)");
         else if(GuardBlocked())
            Print("SP2L: signal skipped (daily guard)");
         else if(EntryOrder == ENTRY_LIMIT)
            PlaceLimit(d, levelB, stop);
         else
            OpenMarket(d, stop);   // we are at the open of the bar after the break
      }
   }

   // 5) adopt a fresh spike when idle
   if(g_dir == 0)
      DetectSpike();
}

//+------------------------------------------------------------------+
//| Candle helpers                                                    |
//+------------------------------------------------------------------+
double BodyRatio(const int shift)
{
   double hi = iHigh(_Symbol, PERIOD_CURRENT, shift);
   double lo = iLow(_Symbol, PERIOD_CURRENT, shift);
   double rng = hi - lo;
   if(rng <= 0.0) return(0.0);
   double o = iOpen(_Symbol, PERIOD_CURRENT, shift);
   double c = iClose(_Symbol, PERIOD_CURRENT, shift);
   return(MathAbs(c - o) / rng);
}

bool IsDoji(const int shift)
{
   return(BodyRatio(shift) <= MaxDojiBodyRatio);
}

// A "strong" candle: directional with a body big enough, and not a doji.
bool IsStrong(const int shift, const int dir)
{
   if(IsDoji(shift)) return(false);
   double o = iOpen(_Symbol, PERIOD_CURRENT, shift);
   double c = iClose(_Symbol, PERIOD_CURRENT, shift);
   if(dir == 1  && c <= o) return(false);
   if(dir == -1 && c >= o) return(false);
   return(BodyRatio(shift) >= MinBodyRatio);
}

//+------------------------------------------------------------------+
//| Spike detection on closed bars: the run ends at shift 1.          |
//+------------------------------------------------------------------+
void DetectSpike()
{
   for(int pass = 0; pass < 2; pass++)
   {
      int dir = (pass == 0) ? 1 : -1;

      // Walk back from shift 1 while candles keep fitting the run.
      int run = 0, strong = 0, dojis = 0;
      for(int s = 1; s <= 300; s++)
      {
         if(IsStrong(s, dir))
         {
            strong++;
            run++;
         }
         else if(IsDoji(s) && dojis < DojiTolerance)
         {
            dojis++;
            run++;
         }
         else
            break;
      }
      if(run < MinSpikeBars) continue;
      if(strong < MinSpikeBars - DojiTolerance) continue;
      if(!IsStrong(1, dir)) continue;          // the last spike candle must be strong

      int startShift = run;                    // oldest bar of the run
      if(RequireGap && !HasGap(run, dir)) continue;

      double pA = (dir == 1) ? iLow(_Symbol, PERIOD_CURRENT, startShift)
                             : iHigh(_Symbol, PERIOD_CURRENT, startShift);
      double pB = (dir == 1) ? iHigh(_Symbol, PERIOD_CURRENT, 1)
                             : iLow(_Symbol, PERIOD_CURRENT, 1);

      if(PowerMult > 0.0)
      {
         double atr = AtrValue();
         if(atr > 0.0 && MathAbs(pB - pA) < PowerMult * atr) continue;
      }

      g_dir          = dir;
      g_pointA       = pA;
      g_pointB       = pB;
      g_spikeEndBars = 0;
      g_inPull       = false;
      g_pullBars     = 0;
      return;
   }
}

// Three-candle imbalance (FVG) anywhere across the run.
bool HasGap(const int run, const int dir)
{
   for(int k = 1; k <= run - 1; k++)
   {
      double lowK  = iLow(_Symbol, PERIOD_CURRENT, k);
      double highK = iHigh(_Symbol, PERIOD_CURRENT, k);
      double high2 = iHigh(_Symbol, PERIOD_CURRENT, k + 2);
      double low2  = iLow(_Symbol, PERIOD_CURRENT, k + 2);
      if(dir == 1  && lowK  > high2) return(true);
      if(dir == -1 && highK < low2)  return(true);
   }
   return(false);
}

double AtrValue()
{
   double buf[];
   if(CopyBuffer(atrHandle, 0, 1, 1, buf) != 1) return(0.0);
   return(buf[0]);
}

void ResetSetup()
{
   g_dir          = 0;
   g_pointA       = 0.0;
   g_pointB       = 0.0;
   g_spikeEndBars = 0;
   g_inPull       = false;
   g_pullBars     = 0;
}

//+------------------------------------------------------------------+
//| Session / guards                                                  |
//+------------------------------------------------------------------+
bool InSession()
{
   if(!UseSession) return(true);
   MqlDateTime now;
   TimeToStruct(TimeCurrent(), now);
   int cur = now.hour * 60 + now.min;
   int s = ParseHHMM(SessionStart);
   int e = ParseHHMM(SessionEnd);
   if(s <= e) return(cur >= s && cur <= e);
   return(cur >= s || cur <= e);   // session spans midnight
}

int ParseHHMM(const string txt)
{
   string parts[];
   if(StringSplit(txt, ':', parts) != 2) return(0);
   return((int)StringToInteger(parts[0]) * 60 + (int)StringToInteger(parts[1]));
}

void ResetDayIfNeeded()
{
   MqlDateTime st;
   TimeToStruct(TimeCurrent(), st);
   st.hour = 0; st.min = 0; st.sec = 0;
   datetime midnight = StructToTime(st);
   if(midnight != g_dayStart)
   {
      g_dayStart     = midnight;
      g_dayTrades    = 0;
      g_dayRiskMoney = AccountInfoDouble(ACCOUNT_EQUITY) * RiskPercent / 100.0;
   }
}

// Realized profit booked today by this EA on this symbol.
double DayRealizedProfit()
{
   if(!HistorySelect(g_dayStart, TimeCurrent())) return(0.0);
   double sum = 0.0;
   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != (long)MagicNumber) continue;
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
      long entryType = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entryType == DEAL_ENTRY_IN) continue;   // only closing deals carry P/L
      sum += HistoryDealGetDouble(ticket, DEAL_PROFIT)
           + HistoryDealGetDouble(ticket, DEAL_SWAP)
           + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
   }
   return(sum);
}

bool GuardBlocked()
{
   if(MaxTradesPerDay > 0 && g_dayTrades >= MaxTradesPerDay) return(true);
   if(MaxDailyLossR > 0.0 && g_dayRiskMoney > 0.0)
   {
      double rToday = DayRealizedProfit() / g_dayRiskMoney;
      if(rToday <= -MaxDailyLossR) return(true);
   }
   return(false);
}

// True when the EA already has a position or a pending order here.
bool Busy()
{
   if(CountPositions() > 0) return(true);
   if(CountOrders() > 0) return(true);
   return(false);
}

int CountPositions()
{
   int n = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Symbol() == _Symbol && posInfo.Magic() == (long)MagicNumber) n++;
   }
   return(n);
}

int CountOrders()
{
   int n = 0;
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!ordInfo.SelectByIndex(i)) continue;
      if(ordInfo.Symbol() == _Symbol && ordInfo.Magic() == (long)MagicNumber) n++;
   }
   return(n);
}

//+------------------------------------------------------------------+
//| Position sizing from the risk percentage and the stop distance    |
//+------------------------------------------------------------------+
double LotsForRisk(const double entry, const double stop)
{
   double slDist = MathAbs(entry - stop);
   if(slDist <= 0.0) return(0.0);

   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tickSize <= 0.0 || tickValue <= 0.0) return(0.0);

   double riskMoney  = AccountInfoDouble(ACCOUNT_EQUITY) * RiskPercent / 100.0;
   double lossPerLot = (slDist / tickSize) * tickValue;
   if(lossPerLot <= 0.0) return(0.0);

   double lots = riskMoney / lossPerLot;

   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(stepLot > 0.0) lots = MathFloor(lots / stepLot) * stepLot;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   return(NormalizeDouble(lots, 2));
}

// Keep stops/targets outside the broker's minimum distance.
double SanitizeStop(const int dir, const double price, const double stop)
{
   long   stopsLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDist    = stopsLevel * _Point;
   double s          = stop;
   if(dir == 1  && price - s < minDist) s = price - minDist;
   if(dir == -1 && s - price < minDist) s = price + minDist;
   return(NormalizeDouble(s, _Digits));
}

//+------------------------------------------------------------------+
//| Order placement                                                   |
//+------------------------------------------------------------------+
void OpenMarket(const int dir, const double stopIn)
{
   double price = (dir == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                            : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double stop  = SanitizeStop(dir, price, stopIn);
   double risk  = MathAbs(price - stop);
   if(risk <= 0.0) return;
   double tp    = (dir == 1) ? price + RiskReward * risk : price - RiskReward * risk;
   tp = NormalizeDouble(tp, _Digits);

   double lots = LotsForRisk(price, stop);
   if(lots <= 0.0)
   {
      Print("SP2L: computed lot size is zero - entry skipped");
      return;
   }

   bool ok = (dir == 1) ? trade.Buy(lots, _Symbol, 0.0, stop, tp, "SP2L")
                        : trade.Sell(lots, _Symbol, 0.0, stop, tp, "SP2L");
   if(ok)
      PrintFormat("SP2L: %s market %.2f lots @ %s SL %s TP %s",
                  (dir == 1 ? "BUY" : "SELL"), lots,
                  DoubleToString(price, _Digits), DoubleToString(stop, _Digits),
                  DoubleToString(tp, _Digits));
   else
      PrintFormat("SP2L: market order failed, retcode=%d", trade.ResultRetcode());
}

void PlaceLimit(const int dir, const double levelB, const double stopIn)
{
   double price = NormalizeDouble(levelB, _Digits);
   double stop  = SanitizeStop(dir, price, stopIn);
   double risk  = MathAbs(price - stop);
   if(risk <= 0.0) return;
   double tp    = (dir == 1) ? price + RiskReward * risk : price - RiskReward * risk;
   tp = NormalizeDouble(tp, _Digits);

   double lots = LotsForRisk(price, stop);
   if(lots <= 0.0)
   {
      Print("SP2L: computed lot size is zero - limit skipped");
      return;
   }

   // A long limit must sit below the ask, a short limit above the bid.
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(dir == 1 && price >= ask)
   {
      Print("SP2L: price already above the long limit level - skipped");
      return;
   }
   if(dir == -1 && price <= bid)
   {
      Print("SP2L: price already below the short limit level - skipped");
      return;
   }

   // Expiry is enforced in ExpireStaleLimits() so it works on every broker.
   bool ok = (dir == 1)
             ? trade.BuyLimit(lots, price, _Symbol, stop, tp, ORDER_TIME_GTC, 0, "SP2L")
             : trade.SellLimit(lots, price, _Symbol, stop, tp, ORDER_TIME_GTC, 0, "SP2L");
   if(ok)
      PrintFormat("SP2L: %s limit %.2f lots @ %s SL %s TP %s",
                  (dir == 1 ? "BUY" : "SELL"), lots,
                  DoubleToString(price, _Digits), DoubleToString(stop, _Digits),
                  DoubleToString(tp, _Digits));
   else
      PrintFormat("SP2L: limit order failed, retcode=%d", trade.ResultRetcode());
}

// Cancel limits that never got filled inside the waiting window.
void ExpireStaleLimits()
{
   int barSeconds = PeriodSeconds(PERIOD_CURRENT);
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!ordInfo.SelectByIndex(i)) continue;
      if(ordInfo.Symbol() != _Symbol) continue;
      if(ordInfo.Magic() != (long)MagicNumber) continue;
      ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)ordInfo.OrderType();
      if(type != ORDER_TYPE_BUY_LIMIT && type != ORDER_TYPE_SELL_LIMIT) continue;

      long age = (long)(TimeCurrent() - ordInfo.TimeSetup());
      if(age >= (long)LimitWaitBars * barSeconds)
      {
         if(trade.OrderDelete(ordInfo.Ticket()))
            Print("SP2L: limit not activated within the window - cancelled");
      }
   }
}

//+------------------------------------------------------------------+
//| Track entry/initial stop per position so R levels stay stable     |
//+------------------------------------------------------------------+
void SyncPositionStates()
{
   // Add states for new positions.
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Symbol() != _Symbol || posInfo.Magic() != (long)MagicNumber) continue;
      ulong ticket = posInfo.Ticket();
      if(FindState(ticket) >= 0) continue;
      int n = ArraySize(g_pos);
      ArrayResize(g_pos, n + 1);
      g_pos[n].ticket      = ticket;
      g_pos[n].entry       = posInfo.PriceOpen();
      g_pos[n].initSL      = posInfo.StopLoss();
      g_pos[n].partialDone = false;
      g_pos[n].beDone      = false;
      // Counted here so both market entries and limit fills hit the guard.
      g_dayTrades++;
   }
   // Drop states whose position is gone.
   for(int k = ArraySize(g_pos) - 1; k >= 0; k--)
   {
      if(!posInfo.SelectByTicket(g_pos[k].ticket))
      {
         for(int j = k; j < ArraySize(g_pos) - 1; j++)
            g_pos[j] = g_pos[j + 1];
         ArrayResize(g_pos, ArraySize(g_pos) - 1);
      }
   }
}

int FindState(const ulong ticket)
{
   for(int i = 0; i < ArraySize(g_pos); i++)
      if(g_pos[i].ticket == ticket) return(i);
   return(-1);
}

//+------------------------------------------------------------------+
//| Partial take-profit, breakeven and ATR trailing                   |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Symbol() != _Symbol || posInfo.Magic() != (long)MagicNumber) continue;

      ulong ticket = posInfo.Ticket();
      int   si     = FindState(ticket);
      if(si < 0) continue;

      bool   isLong = (posInfo.PositionType() == POSITION_TYPE_BUY);
      int    dir    = isLong ? 1 : -1;
      double entry  = g_pos[si].entry;
      double initSL = g_pos[si].initSL;
      double R      = MathAbs(entry - initSL);
      if(R <= 0.0) continue;

      double px = isLong ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                         : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double curSL = posInfo.StopLoss();
      double curTP = posInfo.TakeProfit();
      double newSL = curSL;

      // Partial take-profit.
      if(UsePartial && !g_pos[si].partialDone)
      {
         double tp1 = entry + dir * PartialRR * R;
         bool reached = isLong ? (px >= tp1) : (px <= tp1);
         if(reached)
         {
            double vol     = posInfo.Volume();
            double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
            double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            double part    = vol * PartialPercent / 100.0;
            if(stepLot > 0.0) part = MathFloor(part / stepLot) * stepLot;
            if(part >= minLot && vol - part >= minLot)
            {
               if(trade.PositionClosePartial(ticket, part))
               {
                  g_pos[si].partialDone = true;
                  PrintFormat("SP2L: partial closed %.2f lots at %sR",
                              part, DoubleToString(PartialRR, 2));
                  if(BeMode == BE_AFTER_PARTIAL && !g_pos[si].beDone)
                  {
                     newSL = entry + dir * BeOffsetR * R;
                     g_pos[si].beDone = true;
                  }
               }
            }
            else
            {
               // Too small to split: mark it done so we don't retry forever.
               g_pos[si].partialDone = true;
               Print("SP2L: partial skipped (volume below the broker minimum)");
            }
         }
      }

      // Breakeven at an R trigger.
      if(BeMode == BE_RR && !g_pos[si].beDone)
      {
         double trig = entry + dir * BeTriggerRR * R;
         bool reached = isLong ? (px >= trig) : (px <= trig);
         if(reached)
         {
            newSL = entry + dir * BeOffsetR * R;
            g_pos[si].beDone = true;
         }
      }

      // ATR trailing stop.
      if(TrailAtrMult > 0.0)
      {
         double atr = AtrValue();
         if(atr > 0.0)
         {
            double cand = isLong ? px - TrailAtrMult * atr : px + TrailAtrMult * atr;
            if(isLong  && cand > newSL) newSL = cand;
            if(!isLong && cand < newSL) newSL = cand;
         }
      }

      // Only ever tighten, and respect the broker's minimum distance.
      if(newSL != curSL)
      {
         bool tighter = isLong ? (newSL > curSL) : (newSL < curSL);
         if(tighter)
         {
            double safe = SanitizeStop(dir, px, newSL);
            bool valid = isLong ? (safe < px) : (safe > px);
            if(valid && MathAbs(safe - curSL) > _Point / 2.0)
            {
               if(!trade.PositionModify(ticket, NormalizeDouble(safe, _Digits), curTP))
                  PrintFormat("SP2L: PositionModify failed, retcode=%d", trade.ResultRetcode());
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| On-chart status panel                                             |
//+------------------------------------------------------------------+
void DrawPanel()
{
   string state = (g_dir == 0) ? "IDLE" : (g_inPull ? "PULLBACK" : "SPIKE");
   string dirTxt = (g_dir == 1) ? "LONG" : (g_dir == -1) ? "SHORT" : "-";
   string entryTxt = (EntryOrder == ENTRY_LIMIT) ? "Limit @ B" : "Market next open";
   string breakTxt = (BreakConfirm == BREAK_CLOSE) ? "Close only" : "Shadow or close";
   double rToday = (g_dayRiskMoney > 0.0) ? DayRealizedProfit() / g_dayRiskMoney : 0.0;

   string txt = "SP2L Poursamadi EA\n";
   txt += "State: " + state + "   Dir: " + dirTxt + "\n";
   if(g_dir != 0)
      txt += "B: " + DoubleToString(g_pointB, _Digits)
           + "   A: " + DoubleToString(g_pointA, _Digits) + "\n";
   txt += "Break: " + breakTxt + "   Entry: " + entryTxt + "\n";
   txt += "Positions: " + IntegerToString(CountPositions())
        + "   Orders: " + IntegerToString(CountOrders()) + "\n";
   txt += "Today: " + IntegerToString(g_dayTrades) + " trades   "
        + DoubleToString(rToday, 2) + "R\n";
   Comment(txt);
}
//+------------------------------------------------------------------+
