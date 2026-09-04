"""Reporting: the report must say the uncomfortable things, at the top, in words.

A table of ratios that a reader has to interpret is not the deliverable. The
deliverable is a report whose first screen already answers the question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.benchmark import run_random_benchmark
from src.backtest.engine import run_backtest
from src.config import resolve_path
from src.reporting.summary import StrategyReport, build_markdown, write_report
from src.strategies import RESEARCH_STRATEGIES
from src.strategies.base import Strategy
from tests.conftest import make_bars, make_dataset


def _reports(dataset, cfg, names=("buy_and_hold", "rsi2"), capital=100_000.0, benchmark=False):
    out = []
    for name in names:
        result = run_backtest(dataset, RESEARCH_STRATEGIES[name](), cfg, initial_capital=capital)
        bench = (
            run_random_benchmark(dataset, result, cfg, n_runs=60, seed=3)
            if benchmark and name != "buy_and_hold" else None
        )
        out.append(StrategyReport.build(result, benchmark=bench))
    return out


def test_synthetic_data_is_declared_at_the_very_top(dataset, cfg):
    markdown = build_markdown(_reports(dataset, cfg), dataset, cfg, {})
    banner_position = markdown.index("SYNTHETIC DATA")
    first_table = markdown.index("| Strategy")
    assert banner_position < first_table
    assert "not evidence" in markdown


def test_report_states_plainly_when_costs_exceed_gross_profit(cfg):
    """The specific sentence the spec asks for, in plain language, at the top."""
    class Churn(Strategy):
        name = "churn"

        def generate_signals(self, bars, features):
            values = np.zeros(len(bars), dtype="int8")
            values[::2] = 1
            return pd.Series(values, index=bars.index, dtype="int8")

    # A gentle uptrend: the strategy is right about direction and makes a gross
    # profit, then hands all of it and more to the broker in round trips.
    rng = np.random.default_rng(5)
    closes = 1800 + np.cumsum(rng.normal(0.15, 1.0, 600))
    dataset = make_dataset(make_bars(closes))
    result = run_backtest(dataset, Churn(), cfg, initial_capital=100_000.0)
    assert result.gross_pnl > 0
    assert result.costs_exceed_gross_profit

    markdown = build_markdown([StrategyReport.build(result)], dataset, cfg, {})
    assert "exceeded gross profit" in markdown
    assert markdown.index("exceeded gross profit") < markdown.index("## Out-of-sample performance")


def test_report_shows_the_position_sizing_warning_for_a_small_account(dataset, cfg):
    markdown = build_markdown(_reports(dataset, cfg, capital=50.0), dataset, cfg, {})
    assert "POSITION SIZING WARNING" in markdown
    assert "Minimum viable account size" in markdown
    assert markdown.index("POSITION SIZING WARNING") < markdown.index("## Out-of-sample performance")


def test_report_states_the_minimum_viable_account_size(dataset, cfg):
    markdown = build_markdown(_reports(dataset, cfg, capital=50.0), dataset, cfg, {})
    assert "What this account size can actually do" in markdown
    assert "vs a typical DAILY range" in markdown


def test_report_shows_absolute_usd_beside_percentages(dataset, cfg):
    markdown = build_markdown(_reports(dataset, cfg), dataset, cfg, {})
    assert "Net P&L (USD)" in markdown
    assert "Net return" in markdown
    assert "Gross P&L (USD)" in markdown
    assert "Total costs (USD)" in markdown
    assert "Cost drag (% of gross profit)" in markdown


def test_report_calls_out_a_failure_to_beat_the_random_benchmark(dataset, cfg):
    reports = _reports(dataset, cfg, names=("buy_and_hold", "rsi2"), benchmark=True)
    markdown = build_markdown(reports, dataset, cfg, {})
    assert "random-entry runs" in markdown
    assert "Beats random?" in markdown
    if not reports[1].benchmark.passes:
        assert "no demonstrated edge" in markdown


def test_report_documents_the_fill_rule(dataset, cfg):
    markdown = build_markdown(_reports(dataset, cfg), dataset, cfg, {})
    assert "open of bar t+1" in markdown
    assert "never at the high or the low" in markdown.lower()


def test_write_report_produces_the_markdown_trade_log_and_plots(dataset, cfg, tmp_path):
    reports = _reports(dataset, cfg, names=("rsi2",), benchmark=True)
    path = write_report(reports, dataset, cfg, out_dir=tmp_path / "out")
    assert path.exists()

    out_dir = tmp_path / "out"
    assert (out_dir / "summary.md").exists()
    trade_logs = list(out_dir.glob("trades_*.csv"))
    assert trade_logs, "no trade log was written"

    # The trade log must be inspectable by hand: one row per trade, prices and
    # costs broken out.
    trades = pd.read_csv(trade_logs[0])
    assert len(trades) == reports[0].metrics.n_trades
    for column in ("entry_time", "exit_time", "entry_price_raw", "entry_price_eff",
                   "gross_pnl", "spread_cost", "slippage_cost", "financing_cost",
                   "total_cost", "net_pnl", "r_multiple"):
        assert column in trades.columns

    assert list(out_dir.glob("equity_*.png"))
    assert list(out_dir.glob("drawdown_*.png"))
    assert list(out_dir.glob("random_benchmark_*.png"))


def test_report_lists_every_warning_raised(cfg):
    class AlwaysLong(Strategy):
        name = "always_long"

        def generate_signals(self, bars, features):
            return pd.Series(1, index=bars.index, dtype="int8")

    bars = make_bars(list(np.linspace(1800, 1700, 80)))
    dataset = make_dataset(bars)
    result = run_backtest(dataset, AlwaysLong(), cfg, initial_capital=20.0)
    markdown = build_markdown([StrategyReport.build(result)], dataset, cfg, {})
    assert "Warnings raised during the run" in markdown
    assert "ACCOUNT RUINED" in markdown


def test_parameter_sensitivity_plot_is_labelled_as_in_sample(tmp_path):
    from src.reporting.plots import plot_parameter_sensitivity

    surface = pd.DataFrame({
        "entry_window": [20, 20, 55, 55],
        "atr_stop_multiple": [2.0, 3.0, 2.0, 3.0],
        "objective": [0.1, 0.2, 0.15, 0.05],
        "net_pnl_usd": [1.0, 2.0, 1.5, 0.5],
        "ruined": [False] * 4,
    })
    path = plot_parameter_sensitivity(surface, title="test", path=tmp_path / "sens.png")
    assert path is not None and path.exists()


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def test_cli_costs_command_runs(capsys, tmp_path):
    import run as cli

    exit_code = cli.main(["--log-level", "ERROR", "costs",
                          "--set", f"data.cache_dir={tmp_path / 'c'}"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Cost model" in out
    assert "break even" in out


def test_cli_backtest_writes_a_report(tmp_path):
    import run as cli

    exit_code = cli.main([
        "--log-level", "ERROR", "backtest",
        "--strategy", "buy_and_hold", "--strategy", "trend_donchian",
        "--start", "2023-01-01", "--end", "2023-03-01",
        "--no-benchmark",
        "--set", f"data.cache_dir={tmp_path / 'cache'}",
        "--set", f"reporting.output_dir={tmp_path / 'reports'}",
        "--set", "macro.enabled=false",
    ])
    assert exit_code == 0
    assert (tmp_path / "reports" / "summary.md").exists()


def test_cli_rejects_an_unknown_strategy(tmp_path):
    import run as cli

    with pytest.raises(SystemExit, match="unknown strategy"):
        cli.main(["backtest", "--strategy", "moon_phase",
                  "--set", f"data.cache_dir={tmp_path / 'c'}"])


def test_cli_override_syntax_is_validated():
    import run as cli

    with pytest.raises(SystemExit, match="key=value"):
        cli.main(["backtest", "--set", "nonsense"])
