"""
backtest.py
───────────
Vectorised EMA crossover backtester and parameter optimiser.

Depends on:
  - po3_engine.StrategyEngine   — data prep and EMA computation
  - performance_metrics         — all metric calculations

Usage:
    from backtest import EMABacktester
    bt = EMABacktester(engine)
    perf, outperf = bt.run()
    print(bt.report())

    opt_params, opt_perf = bt.optimise(
        short_range=(5, 30, 2),
        long_range=(20, 100, 5)
    )
"""

import numpy as np
import pandas as pd
from itertools import product

import config
import performance_metrics as pm
from logger import log


class EMABacktester:
    """
    Vectorised EMA crossover backtester.

    Parameters
    ----------
    engine : StrategyEngine
        Used to fetch and prepare OHLCV + EMA data.
    """

    def __init__(self, engine):
        self.engine           = engine
        self.results          = None   # DataFrame after run()
        self.results_overview = None   # DataFrame after optimise()

    # ── Run Backtest ──────────────────────────────────────────────

    def run(self) -> tuple:
        """
        Execute the vectorised backtest on current EMA parameters.

        Logic:
          position  = +1 when EMA_Short > EMA_Long  (long)
          position  = -1 when EMA_Short < EMA_Long  (short)
          strategy  = position.shift(1) × daily_log_return

        Returns
        -------
        (strategy_perf, outperformance_vs_hold) — both as multipliers
        """
        df = self.engine.prepare_backtest_data()
        df = df.dropna()

        df["position"]  = np.where(df["EMA_Short"] > df["EMA_Long"], 1, -1)
        df["strategy"]  = df["position"].shift(1) * df["returns"]
        df.dropna(inplace=True)
        df["creturns"]  = df["returns"].cumsum().apply(np.exp)   # buy-and-hold
        df["cstrategy"] = df["strategy"].cumsum().apply(np.exp)  # EMA strategy

        self.results = df

        perf    = round(df["cstrategy"].iloc[-1], 6)
        outperf = round(perf - df["creturns"].iloc[-1], 6)

        log.info(
            f"Backtest complete | {config.SYMBOL} EMA "
            f"{self.engine.ema_short}/{self.engine.ema_long} | "
            f"Perf: {perf:.4f}x | Outperf: {outperf:+.4f}x"
        )
        return perf, outperf

    # ── Metrics ───────────────────────────────────────────────────

    def get_metrics(self, risk_free_rate: float = 0.02) -> dict:
        """
        Return the full metrics dict for the last completed run().
        Calls run() automatically if results are not yet available.
        """
        if self.results is None:
            self.run()
        return pm.compute_metrics(
            self.results["strategy"],
            self.results["cstrategy"],
            risk_free_rate=risk_free_rate,
        )

    def report(self) -> str:
        """Return a formatted plain-text performance report."""
        metrics = self.get_metrics()
        return pm.format_report(
            metrics,
            symbol    = config.SYMBOL,
            ema_short = self.engine.ema_short,
            ema_long  = self.engine.ema_long,
        )

    def telegram_report(self) -> str:
        """Return a Markdown-formatted report for Telegram."""
        metrics = self.get_metrics()
        return pm.format_telegram_report(
            metrics,
            symbol    = config.SYMBOL,
            ema_short = self.engine.ema_short,
            ema_long  = self.engine.ema_long,
        )

    # ── Optimisation ─────────────────────────────────────────────

    def optimise(self,
                 short_range: tuple = (5, 30, 2),
                 long_range: tuple  = (20, 100, 5)) -> tuple:
        """
        Grid-search for the best EMA parameter combination.

        Parameters
        ----------
        short_range : (start, stop, step) — range for EMA short period
        long_range  : (start, stop, step) — range for EMA long period

        Returns
        -------
        ((opt_short, opt_long), best_performance_multiplier)

        Side effects:
          - engine.ema_short and engine.ema_long are updated to optimal values
          - self.results_overview is set to a DataFrame of all results
        """
        combos  = list(product(range(*short_range), range(*long_range)))
        results = []
        total   = len(combos)

        log.info(f"Optimisation started | {total} EMA combinations to test…")

        for i, (s, l) in enumerate(combos):
            if s >= l:          # invalid combo — skip
                results.append(-np.inf)
                continue

            self.engine.set_parameters(ema_short=s, ema_long=l)
            try:
                perf, _ = self.run()
                results.append(perf)
            except Exception as e:
                log.warning(f"Optimise error at EMA {s}/{l}: {e}")
                results.append(-np.inf)

            if (i + 1) % 20 == 0:
                log.info(f"  …{i + 1}/{total} combinations tested")

        best_idx     = int(np.argmax(results))
        best_perf    = results[best_idx]
        opt_s, opt_l = combos[best_idx]

        # Apply optimal parameters and run final backtest
        self.engine.set_parameters(ema_short=opt_s, ema_long=opt_l)
        self.run()

        # Store all results for heatmap / inspection
        df = pd.DataFrame(combos, columns=["EMA_Short", "EMA_Long"])
        df["performance"] = results
        df = df[df["performance"] > -np.inf]
        self.results_overview = df

        log.info(
            f"Optimisation complete ✅ | "
            f"Best EMA: {opt_s}/{opt_l} | Performance: {best_perf:.4f}x"
        )

        # Also update config so the bot uses the new params going forward
        config.EMA_SHORT = opt_s
        config.EMA_LONG  = opt_l

        return (opt_s, opt_l), round(best_perf, 6)

    # ── Plotting ──────────────────────────────────────────────────

    def plot(self, save_path: str = "backtest_chart.png"):
        """
        4-panel dark-themed strategy dashboard.
        Requires matplotlib — skipped gracefully if not available.
        """
        if self.results is None:
            log.warning("plot(): run the backtest first.")
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend
        except ImportError:
            log.warning("matplotlib not installed — skipping plot.")
            return

        res = self.results
        plt.style.use("dark_background")
        plt.rcParams.update({"figure.facecolor": "#0d1117", "axes.facecolor": "#161b22"})

        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5]})
        fig.suptitle(
            f"{config.SYMBOL}  —  EMA {self.engine.ema_short}/{self.engine.ema_long} Crossover",
            fontsize=16, fontweight="bold", color="#f0f6fc", y=0.98,
        )

        # Price + EMAs
        ax = axes[0]
        ax.plot(res.index, res["price"],     color="#58a6ff", lw=1,   label="Price")
        ax.plot(res.index, res["EMA_Short"], color="#ffa657", lw=1.5, label=f"EMA {self.engine.ema_short}")
        ax.plot(res.index, res["EMA_Long"],  color="#ff7b72", lw=1.5, label=f"EMA {self.engine.ema_long}")
        ax.fill_between(res.index, res["EMA_Short"], res["EMA_Long"],
                        where=res["EMA_Short"] > res["EMA_Long"], alpha=0.12, color="#3fb950")
        ax.fill_between(res.index, res["EMA_Short"], res["EMA_Long"],
                        where=res["EMA_Short"] < res["EMA_Long"], alpha=0.12, color="#ff7b72")
        ax.legend(fontsize=8); ax.set_ylabel("Price"); ax.grid(True, alpha=0.3)

        # Cumulative returns
        ax2 = axes[1]
        ax2.plot(res.index, res["cstrategy"], color="#3fb950", lw=2,   label="EMA Strategy")
        ax2.plot(res.index, res["creturns"],  color="#8b949e", lw=1.5, ls="--", label="Buy & Hold")
        ax2.axhline(1, color="#30363d", ls=":", lw=1)
        ax2.legend(fontsize=8); ax2.set_ylabel("Cum. Return"); ax2.grid(True, alpha=0.3)

        # Drawdown
        ax3 = axes[2]
        dd = pm.drawdown_series(res["cstrategy"])
        ax3.fill_between(res.index, dd, 0, alpha=0.6, color="#da3633")
        ax3.plot(res.index, dd, color="#ff7b72", lw=0.8)
        ax3.set_ylabel("Drawdown %"); ax3.grid(True, alpha=0.3)

        # Position
        ax4 = axes[3]
        ax4.fill_between(res.index, res["position"], 0,
                         where=res["position"] > 0, color="#3fb950", alpha=0.7, label="Long")
        ax4.fill_between(res.index, res["position"], 0,
                         where=res["position"] < 0, color="#da3633", alpha=0.7, label="Short")
        ax4.set_ylabel("Position"); ax4.set_ylim(-1.5, 1.5)
        ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        log.info(f"Backtest chart saved → {save_path}")
        return save_path

    def plot_heatmap(self, save_path: str = "optimisation_heatmap.png"):
        """Plot optimisation heatmap. Requires results_overview from optimise()."""
        if self.results_overview is None:
            log.warning("plot_heatmap(): run optimise() first.")
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            log.warning("matplotlib not installed — skipping heatmap.")
            return

        pivot = self.results_overview.pivot_table(
            index="EMA_Short", columns="EMA_Long", values="performance"
        )
        vals = pivot.values[np.isfinite(pivot.values)]

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#0d1117")

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=vals.min(), vmax=vals.max())
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=7, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_xlabel("EMA Long"); ax.set_ylabel("EMA Short")
        ax.set_title(f"Optimisation Heatmap — {config.SYMBOL}", fontsize=14)

        try:
            opt_row = list(pivot.index).index(self.engine.ema_short)
            opt_col = list(pivot.columns).index(self.engine.ema_long)
            ax.add_patch(plt.Rectangle(
                (opt_col - 0.5, opt_row - 0.5), 1, 1,
                fill=False, edgecolor="white", lw=2
            ))
            ax.text(opt_col, opt_row, "★", ha="center", va="center",
                    fontsize=12, color="white", fontweight="bold")
        except ValueError:
            pass

        plt.colorbar(im, ax=ax, label="Performance (multiplier)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        log.info(f"Heatmap saved → {save_path}")
        return save_path
