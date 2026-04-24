"""
main.py
───────
Entry point for the EMA Crossover Trading Bot.

Wires all modules together and runs the main trading loop:

    config.py              ← all settings
    logger.py              ← logging setup
    mt5_connector.py       ← MT5 session + data
    ema_crossover.py ← EMA signal generation
    risk_manager.py        ← lot sizing, SL/TP, drawdown guard
    trade_manager.py       ← order execution + position management
    performance_metrics.py ← Sharpe, Sortino, Calmar etc.
    backtest.py            ← vectorised backtester + optimiser
    notifier.py            ← Telegram bot + notifications

Run:
    python main.py
"""

import time
import threading
import schedule

import config
from logger          import log
from mt5_connector   import MT5Connector
from ema_crossover      import StrategyEngine
from risk_manager    import RiskManager
from trade_manager   import TradeManager
from backtest        import EMABacktester
from notifier        import TelegramNotifier


# ═══════════════════════════════════════════════════════════════════
#  SHARED BOT STATE
#  Passed by reference into TelegramNotifier so Telegram commands
#  can start/stop the bot and update counters.
# ═══════════════════════════════════════════════════════════════════
bot_state = {
    "running":      False,   # Set True via /start in Telegram
    "trades_today": 0,
    "total_pnl":    0.0,
    "peak_balance": 0.0,
}


# ═══════════════════════════════════════════════════════════════════
#  BOT TICK  —  called every CHECK_INTERVAL_SEC by the scheduler
# ═══════════════════════════════════════════════════════════════════

def bot_tick(connector: MT5Connector,
             engine: StrategyEngine,
             risk_manager: RiskManager,
             trade_manager: TradeManager,
             notifier: TelegramNotifier):
    """
    One iteration of the main trading loop:
      1. Check drawdown circuit breaker
      2. Update trailing stop on open position (if any)
      3. Get current EMA signal
      4. If opposite signal → close existing position
      5. If signal → open new position
    """
    if not bot_state["running"]:
        return

    # ── Reconnect if session dropped ──────────────────────────────
    if not connector.ensure_connected():
        log.error("bot_tick: MT5 reconnect failed — skipping tick.")
        return

    acct = connector.get_account_info()
    if not acct:
        log.warning("bot_tick: could not fetch account info.")
        return

    equity  = acct["equity"]
    balance = acct["balance"]

    # ── Track peak balance ────────────────────────────────────────
    if equity > bot_state["peak_balance"]:
        bot_state["peak_balance"] = equity

    # ── Drawdown circuit breaker ──────────────────────────────────
    if risk_manager.check_drawdown(bot_state["peak_balance"], equity):
        bot_state["running"] = False
        notifier.notify_drawdown_halt(bot_state["peak_balance"], equity)
        return

    # ── Trailing stop update ──────────────────────────────────────
    if config.TRAILING_SL:
        pos = trade_manager.get_open_position()
        if pos:
            risk_manager.update_trailing_stop(pos)

    # ── Signal ────────────────────────────────────────────────────
    signal = engine.get_signal()
    log.info(f"Tick | Signal: {signal} | Equity: ${equity:.2f} | Balance: ${balance:.2f}")

    if signal == "HOLD":
        return

    pos = trade_manager.get_open_position()

    # ── Opposite signal → close current position ──────────────────
    if pos is not None:
        pos_is_buy = pos.type == 0
        sig_is_buy = signal == "BUY"

        if pos_is_buy != sig_is_buy:
            result = trade_manager.close_trade(pos, comment=f"EMABot-{signal}")
            if result["success"]:
                bot_state["total_pnl"] += pos.profit
                notifier.notify_trade_closed(pos, reason=f"Opposite signal ({signal})")
            else:
                log.error(f"Failed to close position: {result}")
                return   # don't open a new trade if close failed
        else:
            log.info(f"Same direction ({signal}) — holding existing position.")
            return

    # ── Open new position ─────────────────────────────────────────
    result = trade_manager.open_trade(signal)

    if result["success"]:
        bot_state["trades_today"] += 1
        notifier.notify_trade_opened(result)
    else:
        log.error(f"Failed to open trade: {result}")


# ═══════════════════════════════════════════════════════════════════
#  SCHEDULER THREAD  —  runs bot_tick in the background
# ═══════════════════════════════════════════════════════════════════

def run_scheduler(connector, engine, risk_manager, trade_manager, notifier):
    """Runs in a daemon thread — fires bot_tick every N seconds."""
    interval = config.CHECK_INTERVAL_SEC
    schedule.every(interval).seconds.do(
        bot_tick, connector, engine, risk_manager, trade_manager, notifier
    )
    log.info(f"Scheduler running — scanning every {interval}s")
    while True:
        schedule.run_pending()
        time.sleep(1)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("   EMA CROSSOVER BOT  —  Starting up")
    log.info("=" * 60)

    # ── 1. Connect to MT5 ─────────────────────────────────────────
    connector = MT5Connector()
    if not connector.connect():
        log.error("Cannot connect to MT5. Check config.py credentials.")
        return

    acct = connector.get_account_info()
    bot_state["peak_balance"] = acct.get("balance", 0)
    log.info(
        f"Account: {acct.get('login')} | "
        f"{acct.get('balance', 0):.2f} {acct.get('currency', '')} | "
        f"Server: {acct.get('server', '')}"
    )

    # ── 2. Build modules ──────────────────────────────────────────
    engine        = StrategyEngine(connector)
    risk_manager  = RiskManager(connector)
    trade_manager = TradeManager(connector, risk_manager)
    backtester    = EMABacktester(engine)

    # ── 3. Build Telegram notifier ────────────────────────────────
    notifier = TelegramNotifier(
        connector, engine, backtester, trade_manager, bot_state
    )
    notifier.build()

    # ── 4. Start scheduler in background thread ───────────────────
    sched_thread = threading.Thread(
        target=run_scheduler,
        args=(connector, engine, risk_manager, trade_manager, notifier),
        daemon=True,
    )
    sched_thread.start()
    log.info("Background scheduler thread started.")

    # ── 5. Start Telegram polling (blocking) ─────────────────────
    log.info("Starting Telegram polling — open your bot and type /start")
    try:
        notifier.start_polling()
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        connector.disconnect()
        log.info("Bot shut down cleanly.")


if __name__ == "__main__":
    main()
