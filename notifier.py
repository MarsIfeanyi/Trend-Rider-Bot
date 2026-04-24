"""
notifier.py
───────────
Telegram bot integration — notifications and control panel.

Handles:
  - Sending trade/alert notifications to your Telegram chat
  - All bot commands (/start, /status, /balance, /signal, etc.)
  - Inline keyboard dashboard
  - Starting/stopping the bot from Telegram

Usage:
    from notifier import TelegramNotifier
    notifier = TelegramNotifier(connector, engine, backtester, trade_manager)
    await notifier.send("Trade opened!")
    notifier.start_polling()   # blocks — run in main thread
"""

import asyncio
import threading

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler,
    CallbackQueryHandler, ContextTypes,
)

import config
from logger import log


class TelegramNotifier:
    """
    Manages all Telegram communication.

    Parameters
    ----------
    connector     : MT5Connector
    engine        : StrategyEngine
    backtester    : EMABacktester
    trade_manager : TradeManager
    bot_state     : dict  — shared mutable state from main.py
    """

    def __init__(self, connector, engine, backtester, trade_manager, bot_state: dict):
        self.connector     = connector
        self.engine        = engine
        self.backtester    = backtester
        self.trade_manager = trade_manager
        self.bot_state     = bot_state
        self.app           = None

    # ── Build the Telegram Application ───────────────────────────

    def build(self):
        """Build the Telegram Application and register all handlers."""
        self.app = ApplicationBuilder().token(config.TG_TOKEN).build()

        self.app.add_handler(CommandHandler("start",    self._cmd_start))
        self.app.add_handler(CommandHandler("status",   self._cmd_status))
        self.app.add_handler(CommandHandler("balance",  self._cmd_balance))
        self.app.add_handler(CommandHandler("signal",   self._cmd_signal))
        self.app.add_handler(CommandHandler("trades",   self._cmd_trades))
        self.app.add_handler(CommandHandler("backtest", self._cmd_backtest))
        self.app.add_handler(CommandHandler("optimize", self._cmd_optimize))
        self.app.add_handler(CommandHandler("settings", self._cmd_settings))
        self.app.add_handler(CallbackQueryHandler(self._button_handler))

        self.app.post_init = self._on_startup
        log.info("Telegram handlers registered.")
        return self

    def start_polling(self):
        """Start polling (blocking — call from main thread)."""
        if self.app is None:
            self.build()
        log.info("Telegram polling started.")
        self.app.run_polling()

    # ── Notification Helpers ──────────────────────────────────────

    async def send(self, text: str, parse_mode: str = "Markdown"):
        """Send a message to the configured chat."""
        try:
            await self.app.bot.send_message(
                chat_id    = config.TG_CHAT_ID,
                text       = text,
                parse_mode = parse_mode,
            )
        except Exception as e:
            log.error(f"Telegram send error: {e}")

    def notify(self, text: str):
        """
        Thread-safe synchronous wrapper around send().
        Safe to call from background threads (scheduler, bot loop).
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self.send(text), loop)
            else:
                asyncio.run(self.send(text))
        except Exception as e:
            log.error(f"notify() error: {e}")

    # ── Notification Templates ────────────────────────────────────

    def notify_trade_opened(self, result: dict):
        signal = result.get("signal", "")
        emoji  = "🟢" if signal == "BUY" else "🔴"
        self.notify(
            f"{emoji} *NEW {signal}*\n"
            f"Symbol: `{result.get('symbol')}`\n"
            f"Entry:  `{result.get('price', 0):.5f}` | Lots: `{result.get('lot')}`\n"
            f"SL: `{result.get('sl', 0):.5f}` | TP: `{result.get('tp', 0):.5f}`\n"
            f"Ticket: `{result.get('ticket')}`"
        )

    def notify_trade_closed(self, position, reason: str = ""):
        direction = "BUY" if position.type == 0 else "SELL"
        emoji     = "🟢" if position.profit >= 0 else "🔴"
        self.notify(
            f"🔄 *POSITION CLOSED*\n"
            f"Symbol: `{position.symbol}` | {direction}\n"
            f"P&L: {emoji} `${position.profit:.2f}`\n"
            f"Ticket: `{position.ticket}`"
            + (f"\nReason: {reason}" if reason else "")
        )

    def notify_drawdown_halt(self, peak: float, equity: float):
        self.notify(
            f"🚨 *MAX DRAWDOWN REACHED — BOT HALTED*\n"
            f"Peak equity: `${peak:.2f}`\n"
            f"Current equity: `${equity:.2f}`\n"
            f"Drawdown: `{(peak - equity) / peak * 100:.2f}%`\n"
            f"Restart with /start after reviewing your settings."
        )

    def notify_startup(self, acct: dict):
        self.notify(
            f"🚀 *EMA Bot Online*\n"
            f"Symbol: `{config.SYMBOL}` | EMA `{config.EMA_SHORT}/{config.EMA_LONG}`\n"
            f"Balance: `${acct.get('balance', 0):.2f} {acct.get('currency', '')}`\n"
            f"Use /start for the control panel."
        )

    # ── Startup Hook ──────────────────────────────────────────────

    async def _on_startup(self, app):
        acct = self.connector.get_account_info()
        await self.send(
            f"🚀 *EMA Bot Online*\n"
            f"Symbol: `{config.SYMBOL}` | EMA `{config.EMA_SHORT}/{config.EMA_LONG}`\n"
            f"Balance: `${acct.get('balance', 0):.2f} {acct.get('currency', '')}`\n"
            f"Use /start for the control panel."
        )

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _get_message(update: Update):
        """
        Safely extract the message object from an Update regardless of
        whether the update came from a slash command or an inline button press.

        Slash command  → update.message        (not None)
        Inline button  → update.callback_query.message  (update.message is None)
        """
        if update.message is not None:
            return update.message
        if update.callback_query is not None:
            return update.callback_query.message
        return None

    # ── Command Handlers ──────────────────────────────────────────
    # Every handler resolves its reply target via _get_message() so
    # it works identically whether triggered by /command or a button.

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        message = self._get_message(update)
        if message is None:
            return
        kb = [
            [InlineKeyboardButton("▶️ Start Bot",   callback_data="start_bot"),
             InlineKeyboardButton("⏹ Stop Bot",    callback_data="stop_bot")],
            [InlineKeyboardButton("📊 Status",      callback_data="status"),
             InlineKeyboardButton("💰 Balance",     callback_data="balance")],
            [InlineKeyboardButton("📈 Backtest",    callback_data="backtest"),
             InlineKeyboardButton("🔍 Signal",      callback_data="signal")],
            [InlineKeyboardButton("📋 Open Trades", callback_data="trades"),
             InlineKeyboardButton("⚙️ Settings",    callback_data="settings")],
        ]
        await message.reply_text(
            "🤖 *EMA CrossOver Bot*\nChoose an action:",
            reply_markup=InlineKeyboardMarkup(kb),
            parse_mode="Markdown",
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        message = self._get_message(update)
        if message is None:
            return
        acct   = self.connector.get_account_info()
        status = "🟢 Running" if self.bot_state.get("running") else "🔴 Stopped"
        msg = (
            f"*BOT STATUS*\n"
            f"State:        {status}\n"
            f"Symbol:       `{config.SYMBOL}`\n"
            f"EMA:          `{config.EMA_SHORT}/{config.EMA_LONG}`\n"
            f"Balance:      `${acct.get('balance', 0):.2f}`\n"
            f"Equity:       `${acct.get('equity',  0):.2f}`\n"
            f"Trades today: `{self.bot_state.get('trades_today', 0)}`\n"
            f"Total PnL:    `${self.bot_state.get('total_pnl', 0):.2f}`"
        )
        await message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        message = self._get_message(update)
        if message is None:
            return
        acct = self.connector.get_account_info()
        msg = (
            f"💰 *ACCOUNT*\n"
            f"Balance:      `${acct.get('balance',      0):.2f}`\n"
            f"Equity:       `${acct.get('equity',       0):.2f}`\n"
            f"Free Margin:  `${acct.get('free_margin',  0):.2f}`\n"
            f"Open P&L:     `${acct.get('profit',       0):.2f}`\n"
            f"Margin Level: `{acct.get('margin_level',  0):.1f}%`\n"
            f"Leverage:     `1:{acct.get('leverage', 0)}`"
        )
        await message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_signal(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        message = self._get_message(update)
        if message is None:
            return
        sig   = self.engine.get_signal()
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig, "⚪")
        await message.reply_text(
            f"{emoji} *Current Signal*: `{sig}`\n"
            f"Symbol: `{config.SYMBOL}` | EMA `{config.EMA_SHORT}/{config.EMA_LONG}`",
            parse_mode="Markdown",
        )

    async def _cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        message = self._get_message(update)
        if message is None:
            return
        pos = self.trade_manager.get_open_position()
        if pos is None:
            await message.reply_text("📋 No open positions.")
            return
        direction = "BUY" if pos.type == 0 else "SELL"
        msg = (
            f"📋 *OPEN POSITION*\n"
            f"Ticket:      `{pos.ticket}`\n"
            f"Direction:   `{direction}`\n"
            f"Volume:      `{pos.volume}`\n"
            f"Open Price:  `{pos.price_open:.5f}`\n"
            f"SL: `{pos.sl:.5f}` | TP: `{pos.tp:.5f}`\n"
            f"Current P&L: `${pos.profit:.2f}`"
        )
        await message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_backtest(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        message = self._get_message(update)
        if message is None:
            return
        await message.reply_text("⏳ Running backtest, please wait…")
        try:
            self.backtester.run()
            report = self.backtester.telegram_report()
            await message.reply_text(report, parse_mode="Markdown")
        except Exception as e:
            await message.reply_text(f"❌ Backtest error: {e}")

    async def _cmd_optimize(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        message = self._get_message(update)
        if message is None:
            return
        await message.reply_text("⏳ Optimising EMA parameters… (30–60 seconds)")
        try:
            opt, perf = self.backtester.optimise()
            msg = (
                f"✅ *OPTIMISATION COMPLETE*\n"
                f"Best EMA: Short=`{opt[0]}`, Long=`{opt[1]}`\n"
                f"Performance: `{perf:.4f}x`\n"
                f"Config updated — bot will use new params."
            )
            await message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            await message.reply_text(f"❌ Optimise error: {e}")

    async def _cmd_settings(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        message = self._get_message(update)
        if message is None:
            return
        msg = (
            f"⚙️ *CURRENT SETTINGS*\n"
            f"Symbol:       `{config.SYMBOL}`\n"
            f"Timeframe:    `{config.TIMEFRAME}`\n"
            f"EMA Short:    `{config.EMA_SHORT}`\n"
            f"EMA Long:     `{config.EMA_LONG}`\n"
            f"SL pips:      `{config.SL_PIPS}`\n"
            f"TP pips:      `{config.TP_PIPS}`\n"
            f"Lot Size:     `{config.LOT_SIZE}`\n"
            f"Risk %:       `{config.RISK_PER_TRADE_PCT}%`\n"
            f"Max DD %:     `{config.MAX_DRAWDOWN_PCT}%`\n"
            f"Trailing SL:  `{config.TRAILING_SL}`\n"
            f"Trail pips:   `{config.TRAIL_PIPS}`\n"
            f"Scan interval:`{config.CHECK_INTERVAL_SEC}s`"
        )
        await message.reply_text(msg, parse_mode="Markdown")

    # ── Inline Button Handler ─────────────────────────────────────

    async def _button_handler(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data    = query.data
        message = query.message   # always use query.message — never mutate update

        if data == "start_bot":
            self.bot_state["running"] = True
            await message.reply_text("✅ Bot *started*.", parse_mode="Markdown")
        elif data == "stop_bot":
            self.bot_state["running"] = False
            await message.reply_text("⏹ Bot *stopped*.", parse_mode="Markdown")
        elif data == "status":   await self._cmd_status(update, ctx)
        elif data == "balance":  await self._cmd_balance(update, ctx)
        elif data == "backtest": await self._cmd_backtest(update, ctx)
        elif data == "signal":   await self._cmd_signal(update, ctx)
        elif data == "trades":   await self._cmd_trades(update, ctx)
        elif data == "settings": await self._cmd_settings(update, ctx)