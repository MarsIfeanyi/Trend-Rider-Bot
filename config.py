"""
config.py
─────────
Single source of truth for all bot settings.
"""

import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env into os.environ

# ═══════════════════════════════════════════════════════════════════
#  MT5 CREDENTIALS
# ═══════════════════════════════════════════════════════════════════
MT5_LOGIN    = int(os.getenv("MT5_LOGIN"))      # getenv returns string, cast to int
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER   = os.getenv("MT5_SERVER")

# ═══════════════════════════════════════════════════════════════════
#  TRADING PARAMETERS
# ═══════════════════════════════════════════════════════════════════
SYMBOL     = "XAUUSD"
TIMEFRAME  = mt5.TIMEFRAME_H1
EMA_SHORT  = 10
EMA_LONG   = 20
LOT_SIZE   = 0.01
MAGIC      = 202400

# ═══════════════════════════════════════════════════════════════════
#  RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════
SL_PIPS            = 50
TP_PIPS            = 100
TRAILING_SL        = True
TRAIL_PIPS         = 30
MAX_DRAWDOWN_PCT   = 5.0
RISK_PER_TRADE_PCT = 1.0

# ═══════════════════════════════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════════════════════════════
TG_TOKEN   = os.getenv("TG_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

# ═══════════════════════════════════════════════════════════════════
#  BOT CONTROL
# ═══════════════════════════════════════════════════════════════════
CHECK_INTERVAL_SEC = 60
BACKTEST_BARS      = 50000
LOG_FILE           = "bot.log"