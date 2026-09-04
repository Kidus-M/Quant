#!/usr/bin/env python
"""Command line entry point.

    python run.py fetch                      download/generate and cache bars
    python run.py backtest                   run every research strategy, write a report
    python run.py backtest --strategy rsi2   run one
    python run.py walkforward --strategy trend_donchian
    python run.py costs                      what the cost model implies, before any strategy
    python run.py risk                       what the configured account size can do

Phase 2, paper signals only (notifies a human, never places an order):

    python run.py alerts check               verify the Telegram credentials
    python run.py alerts test                send one test message
    python run.py alerts once                one pass, suitable for cron
    python run.py alerts run                 poll on a loop
    python run.py alerts state               show the deduplication state

Everything is driven by config/backtest.yaml. Command line flags override single
config keys so an experiment can be reproduced from the config file alone.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backtest.benchmark import ordinal, run_random_benchmark  # noqa: E402
from src.backtest.costs import CostModel  # noqa: E402
from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.sizing import PositionSizer, format_risk_warning  # noqa: E402
from src.backtest.walkforward import WalkForward, parameter_sensitivity  # noqa: E402
from src.config import load_config, load_configs  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.reporting.summary import StrategyReport, write_report  # noqa: E402
from src.strategies import RESEARCH_STRATEGIES  # noqa: E402

log = logging.getLogger("run")


def _parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, _, value = pair.partition("=")
        overrides[key.strip()] = yaml_scalar(value.strip())
    return overrides


def yaml_scalar(text: str):
    import yaml

    return yaml.safe_load(text)


def _load(args) -> tuple:
    paths = [args.config]
    extra = getattr(args, "alerts_config", None)
    if extra:
        paths.append(extra)
    cfg = load_configs(*paths) if len(paths) > 1 else load_config(paths[0])
    overrides = _parse_overrides(getattr(args, "set", []))
    if getattr(args, "start", None):
        overrides["data.start"] = args.start
    if getattr(args, "end", None):
        overrides["data.end"] = args.end
    if getattr(args, "resolution", None):
        overrides["data.resolution"] = args.resolution
    if getattr(args, "capital", None):
        overrides["account.initial_capital_usd"] = float(args.capital)
    if overrides:
        cfg = cfg.with_overrides(overrides)
        log.info("config overrides: %s", overrides)
    return cfg


def _selected(args) -> list[str]:
    if not getattr(args, "strategy", None):
        return list(RESEARCH_STRATEGIES)
    unknown = [s for s in args.strategy if s not in RESEARCH_STRATEGIES]
    if unknown:
        raise SystemExit(f"unknown strategy {unknown}; available: {sorted(RESEARCH_STRATEGIES)}")
    return list(args.strategy)


# ---------------------------------------------------------------------- #
def cmd_fetch(args) -> int:
    cfg = _load(args)
    dataset = load_dataset(cfg, refresh=args.refresh)
    for line in dataset.describe():
        print(line)
    return 0


def cmd_costs(args) -> int:
    """What the cost model implies before any strategy is involved."""
    cfg = _load(args)
    model = CostModel.from_config(cfg)
    print("Cost model")
    for line in model.describe():
        print(f"  {line}")
    print()
    quiet = pd.Timestamp("2024-01-03 13:00", tz="UTC")
    thin = pd.Timestamp("2024-01-03 23:00", tz="UTC")
    size = cfg.get("sizing.fixed_lots", 0.01) * cfg.get("instrument.contract_size_oz_per_lot", 100.0)
    capital = float(cfg.get("account.initial_capital_usd"))
    for label, ts in (("London/NY hours", quiet), ("Asian session", thin)):
        round_trip = model.round_trip_cost_usd(size, ts)
        print(
            f"  {label}: a {size:.2f} oz round trip costs {round_trip:,.3f} USD "
            f"({round_trip / capital:.2%} of a {capital:,.0f} USD account); price must "
            f"move {model.breakeven_move_usd_per_oz(ts):.3f} USD/oz to break even"
        )
    return 0


def cmd_risk(args) -> int:
    cfg = _load(args)
    dataset = load_dataset(cfg)
    strategy = RESEARCH_STRATEGIES["trend_donchian"]()
    result = run_backtest(dataset, strategy, cfg)
    if result.risk is None:
        print("not enough data to compute risk diagnostics")
        return 1
    for line in format_risk_warning(result.risk):
        print(line)
    return 0


def cmd_backtest(args) -> int:
    names = _selected(args)          # validate names before fetching anything
    cfg = _load(args)
    dataset = load_dataset(cfg, refresh=args.refresh)
    reports: list[StrategyReport] = []

    for name in names:
        strategy = RESEARCH_STRATEGIES[name]()
        log.info("running %s", strategy.describe())
        try:
            result = run_backtest(dataset, strategy, cfg)
        except ValueError as exc:
            log.error("%s could not be run: %s", name, exc)
            continue

        benchmark = None
        if not args.no_benchmark and name != "buy_and_hold":
            benchmark = run_random_benchmark(dataset, result, cfg, n_runs=args.benchmark_runs)

        reports.append(StrategyReport.build(result, benchmark=benchmark))

    if not reports:
        log.error("nothing ran")
        return 1

    path = write_report(reports, dataset, cfg)
    print(f"\nreport: {path}")
    _print_console_summary(reports)
    return 0


def cmd_walkforward(args) -> int:
    names = _selected(args)          # validate names before fetching anything
    cfg = _load(args)
    dataset = load_dataset(cfg, refresh=args.refresh)
    reports: list[StrategyReport] = []

    for name in names:
        strategy_class = RESEARCH_STRATEGIES[name]
        if not getattr(strategy_class, "param_grid", None):
            log.info("%s has no parameter grid; running it once out of sample instead", name)
            result = run_backtest(dataset, strategy_class(), cfg)
            reports.append(StrategyReport.build(result))
            continue

        log.info("walk-forward for %s", name)
        walk = WalkForward(dataset, strategy_class, cfg, objective=args.objective).run()
        stitched = walk.as_backtest_result()
        benchmark = None
        if not args.no_benchmark:
            benchmark = run_random_benchmark(dataset, stitched, cfg, n_runs=args.benchmark_runs)
        sensitivity = (
            parameter_sensitivity(dataset, strategy_class, cfg, objective=args.objective)
            if not args.no_sensitivity else pd.DataFrame()
        )
        reports.append(StrategyReport.build(
            stitched, benchmark=benchmark, walk_forward=walk,
            sensitivity=sensitivity, label=f"{name} (walk-forward OOS)",
        ))

    if not reports:
        log.error("nothing ran")
        return 1
    path = write_report(reports, dataset, cfg)
    print(f"\nreport: {path}")
    _print_console_summary(reports)
    return 0


def _print_console_summary(reports: list[StrategyReport]) -> None:
    print()
    header = f"{'strategy':<34}{'net USD':>12}{'gross':>11}{'costs':>10}{'trades':>8}{'random pct':>12}"
    print(header)
    print("-" * len(header))
    for report in reports:
        bench = report.benchmark
        pct = (
            ordinal(bench.strategy_percentile)
            if bench is not None and bench.n_runs else "-"
        )
        print(
            f"{report.name:<34}{report.metrics.net_pnl_usd:>12,.2f}"
            f"{report.metrics.gross_pnl_usd:>11,.2f}{report.metrics.total_cost_usd:>10,.2f}"
            f"{report.metrics.n_trades:>8,}{pct:>12}"
        )
    print()
    for report in reports:
        for warning in report.result.warnings:
            print(f"  ! {report.name}: {warning}")


# ---------------------------------------------------------------------- #
# Phase 2: paper-signal alerting
# ---------------------------------------------------------------------- #
def _build_runner(args, *, dry_run: bool | None = None):
    from src.alerts.runner import AlertRunner, AlertSettings

    cfg = _load(args)
    settings = AlertSettings.from_config(cfg)
    if dry_run is not None:
        settings.dry_run = dry_run
    if getattr(args, "dry_run", False):
        settings.dry_run = True
    settings.validate()
    return AlertRunner(cfg, settings=settings), cfg, settings


def cmd_alerts(args) -> int:
    from src.alerts.state import AlertStore
    from src.alerts.telegram import TelegramClient, TelegramCredentials, TelegramError

    action = args.action

    if action == "check":
        try:
            credentials = TelegramCredentials.from_env()
            username = TelegramClient(credentials).check_credentials()
        except TelegramError as exc:
            print(f"credentials NOT working: {exc}")
            return 1
        print(f"credentials OK: connected as @{username}")
        print(f"chat id: {credentials.chat_id}")
        return 0

    if action == "state":
        cfg = _load(args)
        store = AlertStore.load(cfg.get("alerts.state_path", "data/alerts_state.json"))
        print(f"state file: {store.path}")
        print(f"checks run: {store.checks}")
        print(f"last check: {_epoch(store.last_check_epoch)}")
        print(f"last heartbeat: {_epoch(store.last_heartbeat_epoch)}")
        print(f"messages in the last hour: {len(store.send_times)}")
        if not store.setups:
            print("no setups tracked yet")
        for key, setup in sorted(store.setups.items()):
            print(
                f"  {key:<34} {setup.status:<8} alerts={setup.alert_count} "
                f"level={setup.last_level} last={_epoch(setup.last_alert_epoch)}"
            )
        return 0

    runner, cfg, settings = _build_runner(args)

    if action == "test":
        from src.alerts.formatter import format_heartbeat

        text = format_heartbeat(
            symbol=settings.symbol_label,
            now=pd.Timestamp.now(tz="UTC"),
            last_bar=None, bars_seen=0, checks=runner.store.checks,
            strategy_states=[f"{name}: configured, not yet evaluated"
                             for name in settings.strategies],
            armed_setups=0, market_open=False,
            evidence_note=settings.evidence_note, is_synthetic=False,
        )
        text = text.replace("Heartbeat</b>", "Test message</b>")
        runner.client.send(text)
        print("test message sent" + (" (dry run)" if settings.dry_run else ""))
        return 0

    if action == "once":
        outcome = runner.check_once()
        _print_alert_outcome(outcome)
        return 1 if outcome.errors and not outcome.sent else 0

    if action == "run":
        runner.run_forever(
            interval_seconds=args.interval,
            max_iterations=args.max_iterations,
        )
        return 0

    raise SystemExit(f"unknown alerts action {action!r}")


def _epoch(value) -> str:
    if not value:
        return "never"
    return pd.Timestamp(value, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")


def _print_alert_outcome(outcome) -> None:
    print(f"last bar: {outcome.last_bar}")
    print(f"market open: {outcome.market_open}   stale data: {outcome.stale}")
    for state in outcome.strategy_states:
        print(f"  {state}")
    if outcome.sent:
        for message in outcome.sent:
            print(f"  sent {message.kind} ({message.key}), delivered={message.delivered}")
    else:
        print("  nothing to send")
    for error in outcome.errors:
        print(f"  error: {error}")


# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/backtest.yaml")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--start")
        p.add_argument("--end")
        p.add_argument("--resolution")
        p.add_argument("--capital", type=float)
        p.add_argument("--set", action="append", metavar="key=value",
                       help="override any config key, e.g. --set costs.slippage_usd_per_oz_per_side=0.2")
        p.add_argument("--refresh", action="store_true", help="ignore the cache and refetch")
        return p

    fetch = common(sub.add_parser("fetch", help="download or generate bars into the cache"))
    fetch.set_defaults(func=cmd_fetch)

    costs = common(sub.add_parser("costs", help="what the cost model implies"))
    costs.set_defaults(func=cmd_costs, refresh=False)

    risk = common(sub.add_parser("risk", help="what the configured account size can do"))
    risk.set_defaults(func=cmd_risk)

    back = common(sub.add_parser("backtest", help="run strategies over the full sample"))
    back.add_argument("--strategy", action="append")
    back.add_argument("--no-benchmark", action="store_true")
    back.add_argument("--benchmark-runs", type=int, default=None)
    back.set_defaults(func=cmd_backtest)

    walk = common(sub.add_parser("walkforward", help="anchored walk-forward validation"))
    walk.add_argument("--strategy", action="append")
    walk.add_argument("--objective", default="sharpe", choices=["sharpe", "net_pnl"])
    walk.add_argument("--no-benchmark", action="store_true")
    walk.add_argument("--no-sensitivity", action="store_true")
    walk.add_argument("--benchmark-runs", type=int, default=None)
    walk.set_defaults(func=cmd_walkforward)

    alerts = common(sub.add_parser(
        "alerts",
        help="phase 2 paper-signal alerting (notifies a human, never places an order)",
    ))
    alerts.add_argument("action", choices=["check", "test", "once", "run", "state"])
    alerts.add_argument("--alerts-config", default="config/alerts.yaml")
    alerts.add_argument("--interval", type=int, default=None,
                        help="seconds between checks in run mode")
    alerts.add_argument("--max-iterations", type=int, default=None,
                        help="stop after this many passes; useful under a supervisor")
    alerts.add_argument("--dry-run", action="store_true",
                        help="format and log messages without contacting Telegram")
    alerts.set_defaults(func=cmd_alerts)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
