"""
trade_manager.py
────────────────
Handles all order execution and live position management:
  - Place market orders (BUY / SELL)
  - Close open positions
  - Query open positions by symbol + magic number
  - Retrieve trade history

Usage:
    from trade_manager import TradeManager
    tm = TradeManager(connector, risk_manager)
    result = tm.open_trade("BUY")
    result = tm.close_trade(position)
    pos    = tm.get_open_position()
"""

import MetaTrader5 as mt5

import config
from logger import log


class TradeManager:
    """
    Executes and manages trades on MT5.

    Parameters
    ----------
    connector    : MT5Connector   — active MT5 session
    risk_manager : RiskManager    — for SL/TP and lot sizing
    """

    def __init__(self, connector, risk_manager):
        self.connector    = connector
        self.risk_manager = risk_manager

    # ── Query Positions ───────────────────────────────────────────

    def get_open_position(self, symbol: str = None, magic: int = None):
        """
        Return this bot's open position for the given symbol, or None.

        Uses config defaults for symbol and magic if not supplied.
        """
        symbol = symbol or config.SYMBOL
        magic  = magic  or config.MAGIC

        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return None
        for p in positions:
            if p.magic == magic:
                return p
        return None

    def get_all_positions(self, symbol: str = None) -> list:
        """Return all open positions for a symbol (all magic numbers)."""
        symbol = symbol or config.SYMBOL
        positions = mt5.positions_get(symbol=symbol)
        return list(positions) if positions else []

    def get_trade_history(self, days_back: int = 30) -> list:
        """Return closed deals from the last N days."""
        from datetime import datetime, timedelta
        date_from = datetime.now() - timedelta(days=days_back)
        deals = mt5.history_deals_get(date_from, datetime.now())
        if deals is None:
            return []
        return [d for d in deals if d.magic == config.MAGIC]

    # ── Open Trade ────────────────────────────────────────────────

    def open_trade(self, signal: str,
                   symbol: str = None,
                   lot: float = None,
                   sl: float = None,
                   tp: float = None,
                   comment: str = "EMABot") -> dict:
        """
        Place a market order in the direction of the signal.

        Parameters
        ----------
        signal  : "BUY" or "SELL"
        symbol  : trading symbol (defaults to config.SYMBOL)
        lot     : lot size — if None, calculated via RiskManager
        sl      : stop-loss price — if None, calculated via RiskManager
        tp      : take-profit price — if None, calculated via RiskManager
        comment : order comment shown in MT5

        Returns
        -------
        dict with keys: success (bool), ticket, price, lot, error
        """
        symbol = symbol or config.SYMBOL

        tick = self.connector.get_tick(symbol)
        if tick is None:
            return {"success": False, "error": "No tick data"}

        price = tick.ask if signal == "BUY" else tick.bid

        # ── Lot sizing ────────────────────────────────────────────
        if lot is None:
            acct = self.connector.get_account_info()
            balance = acct.get("balance", 0)
            lot = self.risk_manager.calculate_lot_size(balance, symbol)

        # ── SL / TP ───────────────────────────────────────────────
        if sl is None or tp is None:
            sl, tp = self.risk_manager.get_sl_tp(signal, price, symbol)

        order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lot,
            "type":         order_type,
            "price":        price,
            "sl":           sl,
            "tp":           tp,
            "deviation":    20,
            "magic":        config.MAGIC,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(
                f"open_trade FAILED | Signal: {signal} | "
                f"Retcode: {result.retcode} | Comment: {result.comment}"
            )
            return {
                "success": False,
                "error":   result.comment,
                "retcode": result.retcode,
            }

        log.info(
            f"Trade opened ✅ | {signal} {lot} lots {symbol} @ {price:.5f} | "
            f"SL: {sl:.5f} | TP: {tp:.5f} | Ticket: {result.order}"
        )
        return {
            "success": True,
            "ticket":  result.order,
            "price":   price,
            "lot":     lot,
            "sl":      sl,
            "tp":      tp,
            "signal":  signal,
            "symbol":  symbol,
        }

    # ── Close Trade ───────────────────────────────────────────────

    def close_trade(self, position, comment: str = "EMABot-Close") -> dict:
        """
        Close an existing open position by its MT5 position object.

        Parameters
        ----------
        position : MT5 position object from positions_get()
        comment  : order comment shown in MT5

        Returns
        -------
        dict with keys: success (bool), ticket, profit, error
        """
        tick = self.connector.get_tick(position.symbol)
        if tick is None:
            return {"success": False, "error": "No tick data"}

        close_type  = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
        close_price = tick.bid            if position.type == 0 else tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       position.symbol,
            "volume":       position.volume,
            "type":         close_type,
            "position":     position.ticket,
            "price":        close_price,
            "deviation":    20,
            "magic":        position.magic,
            "comment":      comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(
                f"close_trade FAILED | Ticket: {position.ticket} | "
                f"Retcode: {result.retcode} | Comment: {result.comment}"
            )
            return {"success": False, "error": result.comment, "retcode": result.retcode}

        log.info(
            f"Trade closed ✅ | Ticket: {position.ticket} | "
            f"P&L: ${position.profit:.2f} | Reason: {comment}"
        )
        return {
            "success": True,
            "ticket":  result.order,
            "profit":  position.profit,
        }
