# XAU/USD Intraday Backtesting Engine

A research backtester for intraday spot gold. Its job is to measure whether a
strategy has real expectancy **after costs**, and to make the honest answer
("it does not") easy to reach.

It places no orders. There is no broker integration, not even a stub.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

python run.py costs               # what the cost model implies, before any strategy
python run.py risk                # what the configured account size can actually do
python run.py backtest            # every strategy, full sample, writes reports/summary.md
python run.py walkforward         # anchored walk-forward, the number that counts
pytest                            # 210 tests, including the lookahead suite
```

Out of the box it runs on **synthetic bars** so everything works with no network
and no credentials. Every report generated from synthetic data carries a banner
saying so. Point `data.adapter` at a real source before believing anything.

Useful flags:

```bash
python run.py backtest --strategy rsi2 --start 2022-01-01 --end 2023-12-31
python run.py backtest --capital 5000
python run.py --log-level DEBUG walkforward --strategy trend_donchian
python run.py backtest --set costs.slippage_usd_per_oz_per_side=0.25
```

`--set key=value` overrides any config key, so an experiment is reproducible from
`config/backtest.yaml` plus the command line.

---

## Data source: which one, and why

| Source | Verdict |
|---|---|
| **Dukascopy** | **Chosen for backtesting.** Free, no account, deepest intraday history, and it serves raw **ticks** rather than only pre-built candles. Ticks matter twice over: the 1-minute bars in the cache are then genuinely source data rather than someone else aggregation, and the measured bid/ask spread turns the most important assumption in the cost model into something checkable against the tape. |
| OANDA v20 practice | Clean candle endpoint and a free key, but shallower history and an account requirement. The better choice for the later live-signal phase, which is not built. |
| Twelve Data | Free tier has 1-minute bars but is rate limited and history is short. Not enough for a multi-year walk-forward. |

Implemented adapters: `synthetic` (default, offline), `csv` / `parquet` (a vendor
dump already on disk), `dukascopy` (live fetch), plus `fred` for the macro series.

> **Honest limitation.** The machine this was built on can reach package
> registries but not `datafeed.dukascopy.com` or `fred.stlouisfed.org` directly,
> so the Dukascopy fetcher has **never been run against the live endpoint**. Its
> decoding arithmetic is unit-tested against payloads packed in the documented
> layout (`tests/test_strategies.py::test_dukascopy_tick_decoder_round_trips`),
> and `fetch` sanity-checks decoded prices against a plausible range and refuses
> to cache anything outside it rather than writing nonsense. But the first real
> `python run.py fetch --set data.adapter=dukascopy` may still need the URL
> pattern or record layout adjusted. The FRED client **has** been run for real:
> `DFII10`, `DTWEXBGS`, `DGS10` and `T10YIE` are cached and used by the
> macro-filtered strategy.

### Data rules the layer enforces

- Bars are stored in **UTC, timezone-aware**. A naive index is rejected outright
  rather than localised with a guess.
- **1-minute is the only resolution ever written to the cache.** `ParquetBarCache`
  refuses to store anything else, so a derived resolution can never be mistaken
  for source data six months later. Higher resolutions are resampled on demand.
- Resampling is **left-closed, left-labelled**, and **no bin ever spans a session
  break** (bars are grouped by `(session, bin)`). Empty bins are dropped, never
  forward-filled. See `src/data/resample.py` for why the label convention is only
  safe in combination with the fill rule.
- The cache is month-partitioned with a manifest, so a range already on disk is
  never refetched. The cache key includes any adapter setting that changes the
  data — omitting that was a real bug during development, where every synthetic
  seed silently replayed the first one.
- Quality checks run on every load. **Structural failures** (duplicate timestamps,
  non-positive prices, high below low, bars stamped when the market is shut) are
  dropped and written to `data/quarantine/` with the reason attached.
  **Suspicious-but-plausible bars** (zero volume, large sigma moves) are flagged,
  reported and *kept* by default — gold really does gap on a CPI print, and
  deleting the bars that hurt is the most flattering thing a backtester can do to
  itself. Both are configurable.

---

## The cost model

On a $50 account this file decides the outcome, so it is pessimistic by default.

| Component | Default | Notes |
|---|---|---|
| Spread | 0.30 USD/oz round trip | Halved per side; widened by UTC hour (up to 3x around the daily rollover) and around configured news timestamps |
| Slippage | 0.10 USD/oz per side | Always against the trade. There is no code path that moves a fill favourably |
| Commission | 0.00 USD/lot per side | Present in the model, off by default |
| Financing | -0.15 long / +0.05 short USD/oz/night | Charged on any position held past 21:00 UTC |

**Fills:** a signal generated on bar *t* fills at the **open of bar t+1**, plus
slippage. Never the signal bar close. Never the high or the low. The shift is
applied in exactly one place (`_target_from_signals` in `src/backtest/engine.py`)
so no strategy can opt out.

**Stops are evaluated on closes and exit at the next open.** No intrabar stop
fills: filling intrabar requires assuming the order in which the high and low were
reached inside the bar, and every such assumption flatters the backtest.

Gross P&L, total cost drag and net P&L are reported separately everywhere. The two
are computed by independent routes — from raw prices minus explicit costs, and
from cost-adjusted effective fill prices — and the engine **raises** if they
disagree by more than a millionth of a dollar. If cost drag exceeds gross profit,
the report says so in plain language at the top.

What the defaults mean in practice:

```
London/NY hours: a 1.00 oz round trip costs 0.500 USD (1.00% of a 50 USD account)
Asian session:   a 1.00 oz round trip costs 0.950 USD (1.90% of a 50 USD account)
```

---

## What a $50 account can actually do

This is the single most important output, so the engine computes it and prints it
where it cannot be missed:

```
==============================================================================
POSITION SIZING WARNING
==============================================================================
Account equity 50.00 USD. One ATR of adverse movement (2.59 USD/oz) against the
configured position of 1.00 oz costs 5.2% of the account.
A typical full daily range (25.66 USD/oz) is 51.3% of the account.
The broker minimum position of 1.00 oz (0.01 lots) alone risks 5.2% of equity per
ATR. This account cannot open the smallest tradeable position without taking risk
far above any sane per-trade limit. Ordinary daily noise, not a bad strategy, is
enough to end it.
Minimum viable account size for a 1.0% risk-per-trade rule with a 2x ATR stop:
519 USD.
Measured against a typical DAILY range rather than an intraday stop, the same rule
needs 2,566 USD.
==============================================================================
```

Related engine behaviour: when net equity reaches zero the account is **stopped
out** — the position is closed at the next open and nothing reopens, which is what
a broker margin call actually does. The report says `ACCOUNT RUINED`, names the
date, and ratios that would be meaningless afterwards are reported as `NaN` rather
than as numbers implying the account still existed.

Every summary shows absolute USD beside every percentage.

---

## Strategies

| Name | What it is |
|---|---|
| `buy_and_hold` | The benchmark. Pays one round trip and carries financing every night, so its net result is not the same as the price change |
| `random_entry` | The null hypothesis. Not in the comparison table — it *is* the bar |
| `rsi2` | Connors RSI(2) mean reversion, adapted to intraday |
| `trend_donchian` | Donchian breakout with an ATR trailing stop |
| `trend_macro_filtered` | The same, but longs only while the 10-year real yield is falling on a 20-print basis, shorts only while it is rising |

**On `rsi2`:** the rules were published in 2008 for daily bars on US equity
indices, a market with a structural long bias and index-level mean reversion.
None of that describes intraday spot gold. Applying them to 5-minute bars changes
the holding period by two orders of magnitude while leaving the cost per trade
untouched, which is exactly the regime where a strategy looks excellent gross and
loses money net. Treat any strong intraday result with suspicion and check it
against the random benchmark first.

### The random benchmark is a first-class output

For each strategy the engine measures its trade count, long/short mix and mean
holding period, then runs **1000 random-entry backtests with that same profile and
the same costs**. If the strategy net return does not sit outside the 95th
percentile of that distribution, the report states it has demonstrated nothing.

```
Random benchmark (1000 runs matched to 138 trades, mean hold 4.0 bars, 45% long):
  strategy net P&L          -52.40 USD
  random median             -50.89 USD
  random 95th percentile    -10.28 USD
  strategy sits at the 17.1th percentile of random -- it does NOT beat the random benchmark.
  88% of random runs wiped out the account entirely.
```

### Adding one

Subclass `Strategy`, set `name`, declare `param_grid`, implement `max_lookback`,
`compute_features` and `generate_signals(bars, features) -> Series` returning
`{-1, 0, +1}`. Register it in `src/strategies/__init__.py`. A strategy never sees
the cost model, the portfolio or a future bar; `generate_signals` is handed only
bars and features, and the test suite re-runs every strategy on truncated data to
prove it.

---

## Validation: walk-forward only

- **No random k-fold splits.** Ever. Time-series data is always ordered.
- **Anchored walk-forward:** the training window starts at the beginning of the
  data and grows; the test window is the period immediately after and rolls
  forward. Parameters are fitted on training data alone.
- **Only the concatenated out-of-sample record is reported as the result.**
  In-sample numbers are recorded per fold for diagnosis, never as the headline.
- **Purge and embargo:** the last `embargo_bars` of each training window are
  dropped so parameter selection is not influenced by bars against the test seam,
  and the test window is preceded by a warm-up region in which trading is
  suppressed. Default embargo is the strategy maximum indicator lookback.
- **Trial counting and deflated Sharpe:** every parameter combination evaluated,
  across every fold, is counted and fed into a Bailey/López de Prado deflated
  Sharpe ratio. Searching harder raises the bar. Raw and deflated are both
  reported, with the trial count beside them.
- **Parameter sensitivity:** the full metric surface over the grid is plotted, and
  labelled *in-sample diagnostic, not a performance claim*. A broad plateau is
  weak evidence something is real; a lonely spike surrounded by losses is evidence
  it is not.

A test asserts that the best of 200 pure-noise strategies does **not** clear the
deflated Sharpe bar.

---

## Lookahead bias

`tests/test_lookahead.py` is the most important file here. It covers each item on
the project checklist explicitly, plus one general guard that subsumes most of
them: **truncation invariance**. If a signal at bar *t* depends only on bars up to
*t*, recomputing it on a truncated series must give identical values.

| Checklist item | Test |
|---|---|
| No signal fills within its own bar | `test_fill_is_next_bar_open`, `test_fill_is_never_the_signal_bar_close` |
| No fill at a high or low | `test_fill_never_uses_high_or_low` |
| No `center=True` | `test_no_centred_windows_in_source` (AST scan) |
| No `.shift(-n)` | `test_no_negative_shifts_in_source` (AST scan) |
| No backfilling | `test_no_backfilling_in_source` |
| FRED lagged ≥ 1 day | `test_macro_is_lagged_by_at_least_one_day`, `test_macro_join_never_reaches_forward` |
| Left-closed left-labelled bins | `test_resample_bins_are_left_closed_left_labelled` |
| Normalisation fitted on train only | `test_normaliser_*` |
| Fixed symbol universe | `test_symbol_universe_is_a_single_fixed_symbol` |

**The guards are proved to work.** `src/strategies/cheating.py` contains three
strategies that peek on purpose — a one-bar lookahead, a centred moving average,
and a full-sample z-score — and the suite asserts every one of them is caught. If
those tests ever start passing, the guards are broken and every result the engine
has produced is suspect. The cheats live in a separate registry so they can never
appear in a comparison table.

---

## Layout

```
config/          YAML strategy and backtest config (no constants live in code)
src/
  config.py      dotted-key config with no silent defaults
  data/          sessions, resampling, quality, cache, one module per source
  features/      causal indicators, fit/transform normalisation
  strategies/    one file per strategy, common base, plus the deliberate cheats
  backtest/      cost model, sizing, portfolio, event loop, fast path, walk-forward
  metrics/       performance stats, probabilistic and deflated Sharpe
  reporting/     markdown summary, equity/drawdown/sensitivity plots
tests/           210 tests
notebooks/       exploration only, nothing importable
data/            parquet cache and quarantine (gitignored)
reports/         generated output (gitignored)
```

### Two P&L implementations, and why that is safe

The event loop in `src/backtest/engine.py` is the reference implementation. A
vectorised evaluator in `src/backtest/fast.py` exists only because the random
benchmark runs a thousand backtests (3.5s instead of minutes). A second P&L
implementation is a liability unless it is pinned to the first, so
`tests/test_fast_path.py` asserts the two agree **to the cent** across random
signal paths at five account sizes, including reversals, financing, end-of-data
liquidation and stop-outs. Reproducing the stop-out exactly required modelling all
three moments the engine inspects equity within a bar. The fast path supports
fixed sizing only and refuses anything else rather than approximating it.

---

## Deliberately not built

- **Live order placement.** Out of scope, and not stubbed.
- **Machine learning.** Rules-based strategies only in version one.
- **Phase 2, Telegram alerting.** The spec gates this on phase 1 passing: a
  strategy demonstrating out-of-sample expectancy above the random benchmark net
  of costs. No strategy here has done that, so the alerting is not built. Building
  it now would mean shipping notifications for a signal with no demonstrated edge.
  `.env.example` reserves `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`, and `.env`
  is gitignored, so nothing has to be restructured when the gate is met.

## Known limitations

- The Dukascopy fetcher is unverified against the live endpoint (see above).
- Position size is fixed for the life of a trade. Pyramiding would need a richer
  strategy contract than `{-1, 0, +1}` and is left out rather than half-built.
- The synthetic generator is a driftless random walk with fat tails, volatility
  seasonality, jumps and session gaps. It is useful precisely because it has no
  edge to find — a strategy that profits on it is revealing a bug — but it is not
  a market simulator.
- Financing is a flat per-night rate, not a term structure, and there is no
  triple-swap Wednesday convention.
- Slippage is a constant per side, not a function of size or volatility. For 0.01
  lots that is reasonable; for size it would understate.
