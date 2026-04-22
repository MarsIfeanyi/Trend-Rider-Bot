"""
risk_manager.py
───────────────
All risk and position management logic:
  - Dynamic lot size calculation (fixed fractional)
  - Stop Loss / Take Profit price computation
  - Trailing stop update
  - Drawdown circuit breaker check

Usage:
    from risk_manager import RiskManager
    rm = RiskManager(connector)
    lot      = rm.calculate_lot_size(balance)
    sl, tp   = rm.get_sl_tp("BUY", entry_price)
    halted   = rm.check_drawdown(peak, equity)
    rm.update_trailing_stop(position)
"""

import MetaTrader5 as mt5

import config
from logger import log


class RiskManager:
    """
    Encapsulates all risk management calculations and MT5 SL/TP modifications.

    Parameters
    ----------
    connector : MT5Connector
        Used to read pip values and symbol info.
    """

    def __init__(self, connector):
        self.connector = connector

    # ── Position Sizing ───────────────────────────────────────────

    def calculate_lot_size(self, balance: float,
                           symbol: str = None,
                           sl_pips: int = None,
                           risk_pct: float = None) -> float:
        """
        Fixed-fractional position sizing.

        Formula:
            risk_amount = balance × (risk_pct / 100)
            pip_cost    = pip_value × 10   (per standard lot, approximate)
            lots        = risk_amount / (sl_pips × pip_cost)

        The result is clamped to the broker's minimum lot and rounded
        to the nearest valid lot step.

        Parameters
        ----------
        balance   : float  — current account balance
        symbol    : str    — trading symbol (defaults to config.SYMBOL)
        sl_pips   : int    — stop-loss distance in pips (defaults to config.SL_PIPS)
        risk_pct  : float  — % of balance to risk (defaults to config.RISK_PER_TRADE_PCT)

        Returns
        -------
        float — calculated lot size
        """
        symbol   = symbol   or config.SYMBOL
        sl_pips  = sl_pips  or config.SL_PIPS
        risk_pct = risk_pct or config.RISK_PER_TRADE_PCT

        pip      = self.connector.pip_value(symbol)
        risk_amt = balance * (risk_pct / 100)
        pip_cost = pip * 10
        raw_lots = risk_amt / max(sl_pips * pip_cost, 1e-9)

        sym_info  = self.connector.get_symbol_info(symbol)
        min_lot   = sym_info.volume_min  if sym_info else 0.01
        lot_step  = sym_info.volume_step if sym_info else 0.01
        max_lot   = sym_info.volume_max  if sym_info else 100.0

        lots = round(raw_lots / lot_step) * lot_step
        lots = max(lots, min_lot)
        lots = min(lots, max_lot)

        log.info(
            f"Position sizing | Balance: {balance:.2f} | "
            f"Risk: {risk_pct}% (${risk_amt:.2f}) | "
            f"SL: {sl_pips} pips | Lots: {lots}"
        )
        return round(lots, 2)

    # ── SL / TP Levels ────────────────────────────────────────────

    def get_sl_tp(self, order_type: str, entry_price: float,
                  symbol: str = None,
                  sl_pips: int = None,
                  tp_pips: int = None) -> tuple:
        """
        Convert pip distances into absolute SL and TP price levels.

        Parameters
        ----------
        order_type  : "BUY" or "SELL"
        entry_price : float — the fill / ask / bid price
        symbol      : str   — defaults to config.SYMBOL
        sl_pips     : int   — defaults to config.SL_PIPS
        tp_pips     : int   — defaults to config.TP_PIPS

        Returns
        -------
        (sl_price, tp_price) — tuple of floats
        """
        symbol  = symbol  or config.SYMBOL
        sl_pips = sl_pips or config.SL_PIPS
        tp_pips = tp_pips or config.TP_PIPS

        pip    = self.connector.pip_value(symbol)
        sl_d   = sl_pips * pip
        tp_d   = tp_pips * pip

        if order_type == "BUY":
            sl = round(entry_price - sl_d, 5)
            tp = round(entry_price + tp_d, 5)
        else:  # SELL
            sl = round(entry_price + sl_d, 5)
            tp = round(entry_price - tp_d, 5)

        log.info(f"SL/TP | {order_type} @ {entry_price:.5f} → SL: {sl:.5f} | TP: {tp:.5f}")
        return sl, tp

    # ── Trailing Stop ─────────────────────────────────────────────

    def update_trailing_stop(self, position,
                             trail_pips: int = None,
                             symbol: str = None) -> bool:
        """
        Move the stop-loss to trail price by trail_pips if it
        results in an improvement (tighter SL for BUY, looser for SELL).

        Parameters
        ----------
        position   : MT5 position object from positions_get()
        trail_pips : int — trailing distance in pips (defaults to config.TRAIL_PIPS)
        symbol     : str — defaults to config.SYMBOL

        Returns
        -------
        bool — True if the SL was successfully modified
        """
        trail_pips = trail_pips or config.TRAIL_PIPS
        symbol     = symbol     or config.SYMBOL

        pip  = self.connector.pip_value(symbol)
        tick = self.connector.get_tick(symbol)
        if tick is None:
            return False

        if position.type == 0:  # BUY — trail below bid
            new_sl = round(tick.bid - trail_pips * pip, 5)
            if new_sl <= position.sl:
                return False   # no improvement
        else:                   # SELL — trail above ask
            new_sl = round(tick.ask + trail_pips * pip, 5)
            if position.sl != 0 and new_sl >= position.sl:
                return False   # no improvement

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   symbol,
            "sl":       new_sl,
            "tp":       position.tp,
            "position": position.ticket,
        }
        result = mt5.order_send(request)
        success = result.retcode == mt5.TRADE_RETCODE_DONE

        if success:
            log.info(f"Trailing stop updated | Ticket: {position.ticket} | New SL: {new_sl:.5f}")
        else:
            log.warning(f"Trailing stop update failed | {result.comment}")

        return success

    # ── Drawdown Guard ────────────────────────────────────────────

    def check_drawdown(self, peak_balance: float,
                       current_equity: float,
                       max_dd_pct: float = None) -> bool:
        """
        Return True if the current drawdown has breached the maximum
        allowed threshold.

        Parameters
        ----------
        peak_balance   : float — highest recorded equity
        current_equity : float — current account equity
        max_dd_pct     : float — halt threshold % (defaults to config.MAX_DRAWDOWN_PCT)
        """
        max_dd_pct = max_dd_pct or config.MAX_DRAWDOWN_PCT
        if peak_balance <= 0:
            return False
        drawdown_pct = (peak_balance - current_equity) / peak_balance * 100
        if drawdown_pct >= max_dd_pct:
            log.error(
                f"Drawdown limit breached! | Peak: {peak_balance:.2f} | "
                f"Equity: {current_equity:.2f} | DD: {drawdown_pct:.2f}%"
            )
            return True
        return False
