"""
╔══════════════════════════════════════════════════════════════════╗
║         EMA CROSSOVER TRADING BOT — MT5 + TELEGRAM              ║
║  Features: EMA Strategy | Risk Management | Performance Metrics  ║
║            Sharpe | Sortino | Calmar | SL/TP | Notifications     ║
╚══════════════════════════════════════════════════════════════════╝

DEPENDENCIES:
    pip install MetaTrader5 pandas numpy python-telegram-bot schedule

SETUP:
    1. Fill in CONFIG below with your MT5 + Telegram credentials.
    2. Create a Telegram bot via @BotFather → get TOKEN.
    3. Get your CHAT_ID from @userinfobot.
    4. Run: python ema_crossover_bot.py
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta
import time
import schedule
import asyncio
import threading
import logging

# ── Telegram ──────────────────────────────────────────────────────
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    ContextTypes
)

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← Edit this section before running
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    # ── MT5 Credentials ──────────────────────────────────────────
    "MT5_LOGIN":    12345678,          # Your MT5 account number
    "MT5_PASSWORD": "your_password",   # MT5 password
    "MT5_SERVER":   "Broker-Server",   # e.g. "ICMarkets-Demo"

    # ── Trading Parameters ────────────────────────────────────────
    "SYMBOL":       "XAUUSD",          # Instrument (Gold, EURUSD, etc.)
    "TIMEFRAME":    mt5.TIMEFRAME_H1,  # H1, H4, D1, M15, etc.
    "EMA_SHORT":    9,                 # Short EMA period
    "EMA_LONG":     21,                # Long EMA period
    "LOT_SIZE":     0.01,              # Trade size in lots
    "MAGIC":        202400,            # Unique ID for this bot's orders

    # ── Risk Management ───────────────────────────────────────────
    "SL_PIPS":      50,                # Stop-loss in pips
    "TP_PIPS":      100,               # Take-profit in pips (2:1 R:R)
    "TRAILING_SL":  True,              # Enable trailing stop-loss
    "TRAIL_PIPS":   30,                # Trail distance in pips
    "MAX_DRAWDOWN_PCT": 5.0,           # Halt bot if drawdown > X%
    "RISK_PER_TRADE_PCT": 1.0,         # Risk X% of balance per trade

    # ── Telegram ─────────────────────────────────────────────────
    "TG_TOKEN":     "YOUR_BOT_TOKEN",  # From @BotFather
    "TG_CHAT_ID":   "YOUR_CHAT_ID",    # From @userinfobot

    # ── Bot Control ───────────────────────────────────────────────
    "CHECK_INTERVAL_SEC": 60,          # How often to scan for signals
    "BACKTEST_BARS":      500,         # Bars used for backtesting
    "LOG_FILE":           "bot.log",
}

# ═══════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"]),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════
bot_state = {
    "running": False,
    "trades_today": 0,
    "total_pnl": 0.0,
    "peak_balance": 0.0,
    "tg_app": None,
}

# ═══════════════════════════════════════════════════════════════════
#  MT5 CONNECTION HELPERS
# ═══════════════════════════════════════════════════════════════════

def connect_mt5() -> bool:
    """Initialize and log in to MT5."""
    if not mt5.initialize():
        log.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    authorized = mt5.login(
        CONFIG["MT5_LOGIN"],
        password=CONFIG["MT5_PASSWORD"],
        server=CONFIG["MT5_SERVER"]
    )
    if not authorized:
        log.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    log.info("✅ MT5 connected successfully.")
    return True


def disconnect_mt5():
    mt5.shutdown()
    log.info("MT5 disconnected.")


def get_account_info() -> dict:
    """Return account balance, equity, margin level."""
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        "balance":       info.balance,
        "equity":        info.equity,
        "margin":        info.margin,
        "free_margin":   info.margin_free,
        "margin_level":  info.margin_level,
        "profit":        info.profit,
        "currency":      info.currency,
    }


# ═══════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════

def fetch_ohlcv(symbol: str, timeframe: int, bars: int) -> pd.DataFrame:
    """Fetch OHLCV bars from MT5 into a DataFrame."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        log.error(f"No data for {symbol}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"close": "price"}, inplace=True)
    df["returns"] = np.log(df["price"] / df["price"].shift(1))
    return df


# ═══════════════════════════════════════════════════════════════════
#  EMA STRATEGY ENGINE
# ═══════════════════════════════════════════════════════════════════

class EMABacktester:
    """
    Vectorized EMA crossover backtester with full performance
    and risk metrics.
    """

    def __init__(self, symbol: str, ema_short: int, ema_long: int,
                 start: str, end: str, timeframe=mt5.TIMEFRAME_D1):
        self.symbol    = symbol
        self.ema_short = ema_short
        self.ema_long  = ema_long
        self.start     = start
        self.end       = end
        self.timeframe = timeframe
        self.results   = None
        self.results_overview = None
        self._load_data()

    def __repr__(self):
        return (f"EMABacktester(symbol={self.symbol}, "
                f"EMA_Short={self.ema_short}, EMA_Long={self.ema_long}, "
                f"{self.start} → {self.end})")

    # ── Data ────────────────────────────────────────────────────
    def _load_data(self):
        bars = CONFIG["BACKTEST_BARS"]
        df = fetch_ohlcv(self.symbol, self.timeframe, bars)
        if df.empty:
            raise RuntimeError("Failed to load data from MT5.")
        self.data = df
        self._prepare()

    def _prepare(self):
        d = self.data.copy()
        d["EMA_Short"] = d["price"].ewm(span=self.ema_short, adjust=False).mean()
        d["EMA_Long"]  = d["price"].ewm(span=self.ema_long,  adjust=False).mean()
        self.data = d

    # ── Parameters ──────────────────────────────────────────────
    def set_parameters(self, ema_short=None, ema_long=None):
        if ema_short is not None:
            self.ema_short = ema_short
            self.data["EMA_Short"] = self.data["price"].ewm(
                span=self.ema_short, adjust=False).mean()
        if ema_long is not None:
            self.ema_long = ema_long
            self.data["EMA_Long"] = self.data["price"].ewm(
                span=self.ema_long, adjust=False).mean()

    # ── Backtest ────────────────────────────────────────────────
    def test_strategy(self) -> tuple:
        d = self.data.copy().dropna()
        d["position"]  = np.where(d["EMA_Short"] > d["EMA_Long"], 1, -1)
        d["strategy"]  = d["position"].shift(1) * d["returns"]
        d.dropna(inplace=True)
        d["creturns"]  = d["returns"].cumsum().apply(np.exp)
        d["cstrategy"] = d["strategy"].cumsum().apply(np.exp)
        self.results = d

        perf   = d["cstrategy"].iloc[-1]
        outperf = perf - d["creturns"].iloc[-1]
        return round(perf, 6), round(outperf, 6)

    # ── Performance Metrics ─────────────────────────────────────
    def get_metrics(self, risk_free_rate: float = 0.02) -> dict:
        """
        Returns a full metrics dict:
          - Total Return, CAGR
          - Sharpe Ratio
          - Sortino Ratio
          - Calmar Ratio
          - Max Drawdown
          - Win Rate, Profit Factor
          - Avg Trade, Best/Worst Trade
        """
        if self.results is None:
            self.test_strategy()

        strat = self.results["strategy"].dropna()
        creturns = self.results["cstrategy"]

        # ── Returns ─────────────────────────────────────────────
        total_return = creturns.iloc[-1] - 1
        n_years = len(strat) / 252
        cagr = (creturns.iloc[-1] ** (1 / max(n_years, 0.01))) - 1

        # ── Sharpe Ratio ─────────────────────────────────────────
        daily_rf = risk_free_rate / 252
        excess   = strat - daily_rf
        sharpe   = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() != 0 else 0

        # ── Sortino Ratio ────────────────────────────────────────
        downside = strat[strat < 0]
        downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-9
        sortino  = (strat.mean() * 252 - risk_free_rate) / downside_std

        # ── Max Drawdown ─────────────────────────────────────────
        roll_max   = creturns.cummax()
        drawdown   = (creturns - roll_max) / roll_max
        max_dd     = drawdown.min()

        # ── Calmar Ratio ─────────────────────────────────────────
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # ── Trade Stats ──────────────────────────────────────────
        trades      = strat[strat != 0]
        wins        = trades[trades > 0]
        losses      = trades[trades < 0]
        win_rate    = len(wins) / len(trades) * 100 if len(trades) > 0 else 0
        profit_factor = (wins.sum() / abs(losses.sum())
                         if len(losses) > 0 and losses.sum() != 0 else 0)

        return {
            "Total Return (%)":   round(total_return * 100, 2),
            "CAGR (%)":           round(cagr * 100, 2),
            "Sharpe Ratio":       round(sharpe, 3),
            "Sortino Ratio":      round(sortino, 3),
            "Calmar Ratio":       round(calmar, 3),
            "Max Drawdown (%)":   round(max_dd * 100, 2),
            "Win Rate (%)":       round(win_rate, 2),
            "Profit Factor":      round(profit_factor, 3),
            "Total Trades":       len(trades),
            "Avg Trade (log)":    round(trades.mean(), 6) if len(trades) > 0 else 0,
            "Best Trade (log)":   round(trades.max(), 6) if len(trades) > 0 else 0,
            "Worst Trade (log)":  round(trades.min(), 6) if len(trades) > 0 else 0,
        }

    def metrics_report(self) -> str:
        m = self.get_metrics()
        lines = ["📊 *BACKTEST PERFORMANCE REPORT*",
                 f"Symbol: `{self.symbol}` | EMA {self.ema_short}/{self.ema_long}",
                 "─" * 38]
        for k, v in m.items():
            lines.append(f"  {k:<22} {v}")
        return "\n".join(lines)

    # ── Optimisation ────────────────────────────────────────────
    def optimize(self, short_range: tuple, long_range: tuple) -> tuple:
        """Grid-search best EMA combo. Ranges: (start, end, step)."""
        combos  = list(product(range(*short_range), range(*long_range)))
        results = []
        for s, l in combos:
            if s >= l:
                results.append(-np.inf)
                continue
            self.set_parameters(s, l)
            perf, _ = self.test_strategy()
            results.append(perf)

        best_idx  = int(np.argmax(results))
        best_perf = results[best_idx]
        opt_s, opt_l = combos[best_idx]

        self.set_parameters(opt_s, opt_l)
        self.test_strategy()

        df = pd.DataFrame(combos, columns=["EMA_Short", "EMA_Long"])
        df["performance"] = results
        self.results_overview = df

        return (opt_s, opt_l), round(best_perf, 6)


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL GENERATOR  (live)
# ═══════════════════════════════════════════════════════════════════

def get_signal(symbol: str, ema_short: int, ema_long: int,
               timeframe: int, bars: int = 100) -> str:
    """
    Returns 'BUY', 'SELL', or 'HOLD' based on latest EMA cross.
    Looks at the last completed candle to avoid repainting.
    """
    df = fetch_ohlcv(symbol, timeframe, bars)
    if df.empty:
        return "HOLD"
    df["EMA_Short"] = df["price"].ewm(span=ema_short, adjust=False).mean()
    df["EMA_Long"]  = df["price"].ewm(span=ema_long,  adjust=False).mean()

    # Use index -2 (last *closed* candle)
    prev_short = df["EMA_Short"].iloc[-3]
    prev_long  = df["EMA_Long"].iloc[-3]
    curr_short = df["EMA_Short"].iloc[-2]
    curr_long  = df["EMA_Long"].iloc[-2]

    if prev_short <= prev_long and curr_short > curr_long:
        return "BUY"
    elif prev_short >= prev_long and curr_short < curr_long:
        return "SELL"
    return "HOLD"


# ═══════════════════════════════════════════════════════════════════
#  RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def pip_value(symbol: str) -> float:
    """Returns pip size for the symbol (0.0001 for FX, 0.01 for JPY, 0.1 for Gold)."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0001
    if "JPY" in symbol:
        return 0.01
    if symbol in ("XAUUSD", "XAGUSD"):
        return 0.1
    return info.point * 10


def calculate_lot_size(balance: float, risk_pct: float,
                       sl_pips: int, symbol: str) -> float:
    """Position sizing based on fixed % risk."""
    pip = pip_value(symbol)
    risk_amount  = balance * (risk_pct / 100)
    pip_cost     = pip * 10  # approximate for standard lot
    raw_lots     = risk_amount / (sl_pips * pip_cost)
    info         = mt5.symbol_info(symbol)
    min_lot      = info.volume_min if info else 0.01
    lot_step     = info.volume_step if info else 0.01
    lots = max(round(raw_lots / lot_step) * lot_step, min_lot)
    return round(lots, 2)


def get_sl_tp(order_type: str, price: float, symbol: str,
              sl_pips: int, tp_pips: int) -> tuple:
    """Compute SL and TP prices from pip distances."""
    pip = pip_value(symbol)
    sl_dist = sl_pips * pip
    tp_dist = tp_pips * pip
    if order_type == "BUY":
        return round(price - sl_dist, 5), round(price + tp_dist, 5)
    else:  # SELL
        return round(price + sl_dist, 5), round(price - tp_dist, 5)


def check_drawdown(peak_balance: float, current_equity: float,
                   max_dd_pct: float) -> bool:
    """Returns True if drawdown limit is breached (bot should halt)."""
    if peak_balance <= 0:
        return False
    dd = (peak_balance - current_equity) / peak_balance * 100
    return dd >= max_dd_pct


# ═══════════════════════════════════════════════════════════════════
#  ORDER EXECUTION
# ═══════════════════════════════════════════════════════════════════

def get_open_position(symbol: str, magic: int):
    """Return existing open position for symbol+magic, or None."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    for p in positions:
        if p.magic == magic:
            return p
    return None


def place_order(signal: str, symbol: str, lot: float,
                sl: float, tp: float, magic: int, comment: str = "EMABot") -> dict:
    """Send a market order to MT5."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "error": "No tick data"}

    order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
    price      = tick.ask if signal == "BUY" else tick.bid

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    symbol,
        "volume":    lot,
        "type":      order_type,
        "price":     price,
        "sl":        sl,
        "tp":        tp,
        "deviation": 20,
        "magic":     magic,
        "comment":   comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": result.comment,
                "retcode": result.retcode}
    return {"success": True, "ticket": result.order,
            "price": price, "lot": lot}


def close_position(position) -> dict:
    """Close an open MT5 position."""
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        return {"success": False, "error": "No tick"}

    close_type  = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if position.type == 0 else tick.ask

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    position.symbol,
        "volume":    position.volume,
        "type":      close_type,
        "position":  position.ticket,
        "price":     close_price,
        "deviation": 20,
        "magic":     position.magic,
        "comment":   "EMABot-Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": result.comment}
    return {"success": True, "ticket": result.order}


def update_trailing_stop(position, trail_pips: int, symbol: str) -> bool:
    """Modify SL to trail price by trail_pips."""
    pip = pip_value(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False

    if position.type == 0:  # BUY
        new_sl = round(tick.bid - trail_pips * pip, 5)
        if new_sl <= position.sl:
            return False  # no improvement
    else:  # SELL
        new_sl = round(tick.ask + trail_pips * pip, 5)
        if new_sl >= position.sl or position.sl == 0:
            return False

    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   symbol,
        "sl":       new_sl,
        "tp":       position.tp,
        "position": position.ticket,
    }
    result = mt5.order_send(request)
    return result.retcode == mt5.TRADE_RETCODE_DONE


# ═══════════════════════════════════════════════════════════════════
#  CORE BOT LOOP
# ═══════════════════════════════════════════════════════════════════

def bot_tick():
    """Single iteration of the bot's main loop."""
    if not bot_state["running"]:
        return

    symbol = CONFIG["SYMBOL"]
    acct   = get_account_info()
    if not acct:
        log.warning("Could not fetch account info.")
        return

    equity  = acct["equity"]
    balance = acct["balance"]

    # ── Update peak balance for drawdown tracking ────────────────
    if equity > bot_state["peak_balance"]:
        bot_state["peak_balance"] = equity

    # ── Drawdown guard ───────────────────────────────────────────
    if check_drawdown(bot_state["peak_balance"], equity, CONFIG["MAX_DRAWDOWN_PCT"]):
        msg = (f"🚨 *MAX DRAWDOWN REACHED*\n"
               f"Peak: ${bot_state['peak_balance']:.2f} | "
               f"Equity: ${equity:.2f}\nBot halted.")
        log.error(msg)
        bot_state["running"] = False
        asyncio.run(send_telegram(msg))
        return

    # ── Trailing stop update ─────────────────────────────────────
    if CONFIG["TRAILING_SL"]:
        pos = get_open_position(symbol, CONFIG["MAGIC"])
        if pos:
            update_trailing_stop(pos, CONFIG["TRAIL_PIPS"], symbol)

    # ── Signal ───────────────────────────────────────────────────
    signal = get_signal(
        symbol, CONFIG["EMA_SHORT"], CONFIG["EMA_LONG"],
        CONFIG["TIMEFRAME"]
    )
    log.info(f"Signal: {signal} | Equity: ${equity:.2f}")

    if signal == "HOLD":
        return

    pos = get_open_position(symbol, CONFIG["MAGIC"])

    # ── Opposite signal → close existing ─────────────────────────
    if pos:
        pos_is_buy  = pos.type == 0
        signal_buy  = signal == "BUY"
        if pos_is_buy != signal_buy:
            res = close_position(pos)
            if res["success"]:
                profit = pos.profit
                bot_state["total_pnl"] += profit
                msg = (f"🔄 *POSITION CLOSED*\n"
                       f"Symbol: `{symbol}` | P&L: `${profit:.2f}`\n"
                       f"Reason: Opposite signal ({signal})")
                log.info(msg)
                asyncio.run(send_telegram(msg))
            else:
                log.error(f"Close failed: {res}")
        else:
            return  # Same direction, keep riding it

    # ── Open new position ────────────────────────────────────────
    tick  = mt5.symbol_info_tick(symbol)
    price = tick.ask if signal == "BUY" else tick.bid
    sl, tp = get_sl_tp(signal, price, symbol, CONFIG["SL_PIPS"], CONFIG["TP_PIPS"])
    lot    = calculate_lot_size(balance, CONFIG["RISK_PER_TRADE_PCT"],
                                CONFIG["SL_PIPS"], symbol)

    res = place_order(signal, symbol, lot, sl, tp, CONFIG["MAGIC"])
    if res["success"]:
        bot_state["trades_today"] += 1
        msg = (f"{'🟢' if signal == 'BUY' else '🔴'} *NEW {signal} SIGNAL*\n"
               f"Symbol: `{symbol}`\n"
               f"Entry: `{res['price']:.5f}` | Lots: `{lot}`\n"
               f"SL: `{sl:.5f}` | TP: `{tp:.5f}`\n"
               f"Ticket: `{res['ticket']}`")
        log.info(msg)
        asyncio.run(send_telegram(msg))
    else:
        log.error(f"Order failed: {res}")


# ═══════════════════════════════════════════════════════════════════
#  TELEGRAM INTEGRATION
# ═══════════════════════════════════════════════════════════════════

async def send_telegram(text: str):
    """Fire-and-forget Telegram notification."""
    try:
        app = bot_state.get("tg_app")
        if app:
            await app.bot.send_message(
                chat_id=CONFIG["TG_CHAT_ID"],
                text=text,
                parse_mode="Markdown"
            )
    except Exception as e:
        log.error(f"Telegram send error: {e}")


# ── Command handlers ─────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("▶️ Start Bot",  callback_data="start_bot"),
         InlineKeyboardButton("⏹ Stop Bot",   callback_data="stop_bot")],
        [InlineKeyboardButton("📊 Status",     callback_data="status"),
         InlineKeyboardButton("💰 Balance",    callback_data="balance")],
        [InlineKeyboardButton("📈 Backtest",   callback_data="backtest"),
         InlineKeyboardButton("🔍 Signal",     callback_data="signal")],
        [InlineKeyboardButton("📋 Open Trades",callback_data="trades"),
         InlineKeyboardButton("⚙️ Settings",   callback_data="settings")],
    ]
    markup = InlineKeyboardMarkup(kb)
    await update.message.reply_text(
        "🤖 *EMA CrossOver Bot*\nChoose an action:",
        reply_markup=markup,
        parse_mode="Markdown"
    )


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    acct = get_account_info()
    status = "🟢 Running" if bot_state["running"] else "🔴 Stopped"
    msg = (f"*BOT STATUS*\n"
           f"State: {status}\n"
           f"Symbol: `{CONFIG['SYMBOL']}`\n"
           f"EMA: `{CONFIG['EMA_SHORT']}/{CONFIG['EMA_LONG']}`\n"
           f"Balance: `${acct.get('balance', 0):.2f}`\n"
           f"Equity: `${acct.get('equity', 0):.2f}`\n"
           f"Trades today: `{bot_state['trades_today']}`\n"
           f"Total PnL: `${bot_state['total_pnl']:.2f}`")
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_balance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    acct = get_account_info()
    msg = (f"💰 *ACCOUNT*\n"
           f"Balance:      `${acct.get('balance', 0):.2f}`\n"
           f"Equity:       `${acct.get('equity', 0):.2f}`\n"
           f"Free Margin:  `${acct.get('free_margin', 0):.2f}`\n"
           f"Open P&L:     `${acct.get('profit', 0):.2f}`\n"
           f"Margin Level: `{acct.get('margin_level', 0):.1f}%`")
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sig = get_signal(CONFIG["SYMBOL"], CONFIG["EMA_SHORT"],
                     CONFIG["EMA_LONG"], CONFIG["TIMEFRAME"])
    emoji = "🟢" if sig == "BUY" else ("🔴" if sig == "SELL" else "⚪")
    await update.message.reply_text(
        f"{emoji} *Current Signal*: `{sig}`\n"
        f"Symbol: `{CONFIG['SYMBOL']}` | "
        f"EMA {CONFIG['EMA_SHORT']}/{CONFIG['EMA_LONG']}",
        parse_mode="Markdown"
    )


async def cmd_trades(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    pos = get_open_position(CONFIG["SYMBOL"], CONFIG["MAGIC"])
    if pos is None:
        await update.message.reply_text("📋 No open positions.")
        return
    direction = "BUY" if pos.type == 0 else "SELL"
    msg = (f"📋 *OPEN POSITION*\n"
           f"Ticket: `{pos.ticket}`\n"
           f"Direction: `{direction}`\n"
           f"Volume: `{pos.volume}`\n"
           f"Open Price: `{pos.price_open:.5f}`\n"
           f"SL: `{pos.sl:.5f}` | TP: `{pos.tp:.5f}`\n"
           f"Current P&L: `${pos.profit:.2f}`")
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Running backtest, please wait...")
    try:
        bt = EMABacktester(
            CONFIG["SYMBOL"], CONFIG["EMA_SHORT"], CONFIG["EMA_LONG"],
            start="", end=""
        )
        bt.test_strategy()
        report = bt.metrics_report()
        await update.message.reply_text(f"```\n{report}\n```", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Backtest error: {e}")


async def cmd_optimize(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Optimizing EMA parameters...")
    try:
        bt = EMABacktester(
            CONFIG["SYMBOL"], CONFIG["EMA_SHORT"], CONFIG["EMA_LONG"],
            start="", end=""
        )
        opt, perf = bt.optimize((5, 30, 2), (20, 100, 5))
        msg = (f"✅ *OPTIMIZATION COMPLETE*\n"
               f"Best EMA: Short=`{opt[0]}`, Long=`{opt[1]}`\n"
               f"Best Performance: `{perf:.4f}`")
        CONFIG["EMA_SHORT"] = opt[0]
        CONFIG["EMA_LONG"]  = opt[1]
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Optimize error: {e}")


async def cmd_settings(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = (f"⚙️ *CURRENT SETTINGS*\n"
           f"Symbol:      `{CONFIG['SYMBOL']}`\n"
           f"EMA Short:   `{CONFIG['EMA_SHORT']}`\n"
           f"EMA Long:    `{CONFIG['EMA_LONG']}`\n"
           f"SL (pips):   `{CONFIG['SL_PIPS']}`\n"
           f"TP (pips):   `{CONFIG['TP_PIPS']}`\n"
           f"Lot Size:    `{CONFIG['LOT_SIZE']}`\n"
           f"Risk %:      `{CONFIG['RISK_PER_TRADE_PCT']}%`\n"
           f"Max DD %:    `{CONFIG['MAX_DRAWDOWN_PCT']}%`\n"
           f"Trailing SL: `{CONFIG['TRAILING_SL']}`")
    await update.message.reply_text(msg, parse_mode="Markdown")


# ── Inline button handler ────────────────────────────────────────

async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "start_bot":
        if not bot_state["running"]:
            bot_state["running"] = True
            await query.message.reply_text("✅ Bot *started*.", parse_mode="Markdown")
        else:
            await query.message.reply_text("⚠️ Bot already running.")

    elif data == "stop_bot":
        bot_state["running"] = False
        await query.message.reply_text("⏹ Bot *stopped*.", parse_mode="Markdown")

    elif data == "status":
        update.message = query.message
        await cmd_status(update, ctx)

    elif data == "balance":
        update.message = query.message
        await cmd_balance(update, ctx)

    elif data == "backtest":
        update.message = query.message
        await cmd_backtest(update, ctx)

    elif data == "signal":
        update.message = query.message
        await cmd_signal(update, ctx)

    elif data == "trades":
        update.message = query.message
        await cmd_trades(update, ctx)

    elif data == "settings":
        update.message = query.message
        await cmd_settings(update, ctx)


# ═══════════════════════════════════════════════════════════════════
#  SCHEDULER  (runs bot_tick on interval in background thread)
# ═══════════════════════════════════════════════════════════════════

def run_scheduler():
    interval = CONFIG["CHECK_INTERVAL_SEC"]
    schedule.every(interval).seconds.do(bot_tick)
    log.info(f"Scheduler started — checking every {interval}s")
    while True:
        schedule.run_pending()
        time.sleep(1)


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("   EMA CROSSOVER BOT  —  Starting up")
    log.info("=" * 60)

    # ── Connect MT5 ─────────────────────────────────────────────
    if not connect_mt5():
        log.error("Cannot connect to MT5. Exiting.")
        return

    acct = get_account_info()
    bot_state["peak_balance"] = acct.get("balance", 0)
    log.info(f"Account: {acct.get('balance', 0):.2f} {acct.get('currency', '')}")

    # ── Build Telegram app ───────────────────────────────────────
    tg_app = (ApplicationBuilder()
              .token(CONFIG["TG_TOKEN"])
              .build())

    tg_app.add_handler(CommandHandler("start",    cmd_start))
    tg_app.add_handler(CommandHandler("status",   cmd_status))
    tg_app.add_handler(CommandHandler("balance",  cmd_balance))
    tg_app.add_handler(CommandHandler("signal",   cmd_signal))
    tg_app.add_handler(CommandHandler("trades",   cmd_trades))
    tg_app.add_handler(CommandHandler("backtest", cmd_backtest))
    tg_app.add_handler(CommandHandler("optimize", cmd_optimize))
    tg_app.add_handler(CommandHandler("settings", cmd_settings))
    tg_app.add_handler(CallbackQueryHandler(button_handler))

    bot_state["tg_app"] = tg_app

    # ── Scheduler in background thread ──────────────────────────
    sched_thread = threading.Thread(target=run_scheduler, daemon=True)
    sched_thread.start()

    # ── Startup notification ─────────────────────────────────────
    async def notify_startup(app):
        await send_telegram(
            f"🚀 *EMA Bot Online*\n"
            f"Symbol: `{CONFIG['SYMBOL']}` | "
            f"EMA {CONFIG['EMA_SHORT']}/{CONFIG['EMA_LONG']}\n"
            f"Balance: `${acct.get('balance',0):.2f}`\n"
            f"Use /start for controls."
        )

    tg_app.post_init = notify_startup

    log.info("Starting Telegram polling...")
    tg_app.run_polling()

    # ── Cleanup ──────────────────────────────────────────────────
    disconnect_mt5()


if __name__ == "__main__":
    main()
