"""
po3_engine.py
─────────────
EMA Crossover Strategy Engine.

Responsibilities:
  - Compute EMA indicators on OHLCV data
  - Generate BUY / SELL / HOLD signals from last closed candle
  - Expose set_parameters() for live parameter updates

The name po3_engine reflects the Power-of-3 (PO3) candle structure
awareness — signals are always read from the last *closed* candle
(index -2), never the live candle, to avoid repainting.

Usage:
    from po3_engine import StrategyEngine
    engine = StrategyEngine(connector)
    signal = engine.get_signal()          # "BUY" | "SELL" | "HOLD"
"""

import numpy as np
import pandas as pd

import config
from logger import log


class StrategyEngine:
    """
    Computes EMA crossover signals and prepares data for backtesting.

    Parameters
    ----------
    connector : MT5Connector
        An active MT5Connector instance used to pull OHLCV data.
    """

    def __init__(self, connector):
        self.connector = connector
        self.ema_short = config.EMA_SHORT
        self.ema_long  = config.EMA_LONG
        self.symbol    = config.SYMBOL
        self.timeframe = config.TIMEFRAME

    # ── Parameter Updates ─────────────────────────────────────────

    def set_parameters(self, ema_short: int = None, ema_long: int = None):
        """Update EMA periods at runtime (used by optimiser)."""
        if ema_short is not None:
            self.ema_short = ema_short
        if ema_long is not None:
            self.ema_long = ema_long

    # ── Indicator Computation ─────────────────────────────────────

    def add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add EMA_Short and EMA_Long columns to a DataFrame.
        Expects a 'price' (close) column.
        """
        df = df.copy()
        df["EMA_Short"] = df["price"].ewm(span=self.ema_short, adjust=False).mean()
        df["EMA_Long"]  = df["price"].ewm(span=self.ema_long,  adjust=False).mean()
        return df

    # ── Live Signal ───────────────────────────────────────────────

    def get_signal(self, bars: int = 100) -> str:
        """
        Fetch live data and return the current signal.

        Reads candles at index -3 and -2 (the last two CLOSED candles)
        to detect a crossover without repainting on the open candle.

        Returns
        -------
        "BUY"  — bullish crossover detected
        "SELL" — bearish crossover detected
        "HOLD" — no crossover, stay flat / hold current position
        """
        df = self.connector.fetch_ohlcv(self.symbol, self.timeframe, bars)
        if df.empty or len(df) < self.ema_long + 3:
            log.warning("get_signal: insufficient data — returning HOLD")
            return "HOLD"

        df = self.add_ema(df)

        prev_s = df["EMA_Short"].iloc[-3]
        prev_l = df["EMA_Long"].iloc[-3]
        curr_s = df["EMA_Short"].iloc[-2]
        curr_l = df["EMA_Long"].iloc[-2]

        if prev_s <= prev_l and curr_s > curr_l:
            log.info(f"Signal: BUY  | EMA{self.ema_short}={curr_s:.5f} crossed above EMA{self.ema_long}={curr_l:.5f}")
            return "BUY"
        elif prev_s >= prev_l and curr_s < curr_l:
            log.info(f"Signal: SELL | EMA{self.ema_short}={curr_s:.5f} crossed below EMA{self.ema_long}={curr_l:.5f}")
            return "SELL"

        return "HOLD"

    # ── Data Preparation for Backtesting ─────────────────────────

    def prepare_backtest_data(self, bars: int = None) -> pd.DataFrame:
        """
        Fetch OHLCV data and attach EMA + position columns.
        Used by EMABacktester in backtest.py.
        """
        bars = bars or config.BACKTEST_BARS
        df   = self.connector.fetch_ohlcv(self.symbol, self.timeframe, bars)
        if df.empty:
            raise RuntimeError(f"Could not fetch data for {self.symbol}")
        df = self.add_ema(df)
        return df
