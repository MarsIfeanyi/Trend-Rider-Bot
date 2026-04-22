"""
config.py
─────────
Single source of truth for all bot settings.
Edit this file to change any parameter — nothing else needs to be touched.
"""

import MetaTrader5 as mt5

# ═══════════════════════════════════════════════════════════════════
#  MT5 CREDENTIALS
# ═══════════════════════════════════════════════════════════════════
MT5_LOGIN    = 12345678          # Your MT5 account number
MT5_PASSWORD = "your_password"   # Your MT5 password
MT5_SERVER   = "Broker-Server"   # e.g. "ICMarkets-Demo02"

# ═══════════════════════════════════════════════════════════════════
#  TRADING PARAMETERS
# ═══════════════════════════════════════════════════════════════════
SYMBOL     = "XAUUSD"            # Exact symbol name from MT5 Market Watch
TIMEFRAME  = mt5.TIMEFRAME_H1   # mt5.TIMEFRAME_M15 / H1 / H4 / D1
EMA_SHORT  = 9                   # Fast EMA period
EMA_LONG   = 21                  # Slow EMA period
LOT_SIZE   = 0.01                # Fallback lot size (overridden by position sizing)
MAGIC      = 202400              # Unique bot order ID — do not change mid-session

# ═══════════════════════════════════════════════════════════════════
#  RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════
SL_PIPS            = 50          # Stop-loss distance in pips
TP_PIPS            = 100         # Take-profit distance in pips (2:1 R:R)
TRAILING_SL        = True        # Enable trailing stop-loss
TRAIL_PIPS         = 30          # Trailing distance behind price in pips
MAX_DRAWDOWN_PCT   = 5.0         # Halt bot if drawdown exceeds this %
RISK_PER_TRADE_PCT = 1.0         # Risk this % of balance per trade

# ═══════════════════════════════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════════════════════════════
TG_TOKEN   = "YOUR_BOT_TOKEN"    # From @BotFather
TG_CHAT_ID = "YOUR_CHAT_ID"      # From @userinfobot

# ═══════════════════════════════════════════════════════════════════
#  BOT CONTROL
# ═══════════════════════════════════════════════════════════════════
CHECK_INTERVAL_SEC = 60          # How often to scan for signals (seconds)
BACKTEST_BARS      = 500         # Number of bars loaded for backtesting
LOG_FILE           = "bot.log"   # Log output file name
