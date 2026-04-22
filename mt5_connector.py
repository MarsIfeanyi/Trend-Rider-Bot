"""
mt5_connector.py
────────────────
Handles everything related to the MetaTrader 5 connection:
  - Initialise / login / shutdown
  - Fetch OHLCV data
  - Fetch account information
  - Fetch live tick prices

Usage:
    from mt5_connector import MT5Connector
    conn = MT5Connector()
    conn.connect()
    df   = conn.fetch_ohlcv("XAUUSD", mt5.TIMEFRAME_H1, 500)
    acct = conn.get_account_info()
    conn.disconnect()
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

import config
from logger import log


class MT5Connector:
    """Manages the MT5 session and all data retrieval."""

    def __init__(self):
        self.connected = False

    # ── Connection ────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialise MT5 and log in with credentials from config."""
        if not mt5.initialize():
            log.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False

        authorized = mt5.login(
            config.MT5_LOGIN,
            password=config.MT5_PASSWORD,
            server=config.MT5_SERVER,
        )
        if not authorized:
            log.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

        self.connected = True
        info = mt5.account_info()
        log.info(
            f"MT5 connected | Account: {info.login} | "
            f"Balance: {info.balance:.2f} {info.currency} | "
            f"Server: {info.server}"
        )
        return True

    def disconnect(self):
        """Cleanly shut down the MT5 connection."""
        mt5.shutdown()
        self.connected = False
        log.info("MT5 disconnected.")

    def ensure_connected(self) -> bool:
        """Reconnect if the session has dropped."""
        if not mt5.terminal_info():
            log.warning("MT5 session lost — attempting reconnect…")
            return self.connect()
        return True

    # ── Account Info ─────────────────────────────────────────────

    def get_account_info(self) -> dict:
        """Return a dict of key account metrics."""
        info = mt5.account_info()
        if info is None:
            log.error(f"get_account_info() failed: {mt5.last_error()}")
            return {}
        return {
            "login":        info.login,
            "name":         info.name,
            "balance":      info.balance,
            "equity":       info.equity,
            "margin":       info.margin,
            "free_margin":  info.margin_free,
            "margin_level": info.margin_level,
            "profit":       info.profit,
            "currency":     info.currency,
            "leverage":     info.leverage,
            "server":       info.server,
        }

    # ── Market Data ───────────────────────────────────────────────

    def fetch_ohlcv(self, symbol: str, timeframe: int, bars: int) -> pd.DataFrame:
        """
        Fetch OHLCV bars from MT5.

        Returns a DataFrame indexed by datetime with columns:
            open, high, low, price (close), tick_volume, returns
        Returns an empty DataFrame on failure.
        """
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            log.error(f"fetch_ohlcv: no data for {symbol} — {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"close": "price"}, inplace=True)
        df["returns"] = np.log(df["price"] / df["price"].shift(1))
        return df

    def get_tick(self, symbol: str):
        """Return the latest tick for a symbol, or None."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            log.error(f"get_tick: no tick for {symbol} — {mt5.last_error()}")
        return tick

    def get_symbol_info(self, symbol: str):
        """Return MT5 symbol_info object, or None."""
        info = mt5.symbol_info(symbol)
        if info is None:
            log.error(f"get_symbol_info: unknown symbol {symbol}")
        return info

    def pip_value(self, symbol: str) -> float:
        """
        Return the pip size for a symbol.
            Gold / Silver  → 0.1
            JPY pairs      → 0.01
            Everything else → point × 10
        """
        if symbol in ("XAUUSD", "XAGUSD"):
            return 0.1
        if "JPY" in symbol:
            return 0.01
        info = mt5.symbol_info(symbol)
        if info:
            return info.point * 10
        return 0.0001
