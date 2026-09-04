"""Cost model and portfolio accounting, against hand-computed cases.

Every number in this file was worked out on paper first. That is the point: if the
cost model drifts, these break, and a cost model that drifts downward is how a
losing strategy starts looking profitable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import BUY, SELL, CostModel, TradeCosts
from src.backtest.portfolio import Portfolio

NORMAL_HOUR = pd.Timestamp("2024-01-03 13:00", tz="UTC")   # multiplier 1.0
ASIAN_HOUR = pd.Timestamp("2024-01-03 23:00", tz="UTC")    # multiplier 2.5 in the shipped config


# ---------------------------------------------------------------------- #
# Spread, slippage, commission
# ---------------------------------------------------------------------- #
def test_half_spread_is_half_the_round_trip(flat_cost_model):
    # 0.30 round trip -> 0.15 per side.
    assert flat_cost_model.half_spread_per_oz(NORMAL_HOUR) == pytest.approx(0.15)


def test_cost_per_side_is_half_spread_plus_slippage(flat_cost_model):
    assert flat_cost_model.cost_per_oz(NORMAL_HOUR) == pytest.approx(0.25)


def test_one_ounce_round_trip_costs_fifty_cents(flat_cost_model):
    assert flat_cost_model.round_trip_cost_usd(1.0, NORMAL_HOUR) == pytest.approx(0.50)


def test_breakeven_move_matches_the_round_trip_cost(flat_cost_model):
    # One ounce must move 0.50 USD just to cover getting in and out.
    assert flat_cost_model.breakeven_move_usd_per_oz(NORMAL_HOUR) == pytest.approx(0.50)


def test_spread_widens_in_the_asian_session(cfg):
    model = CostModel.from_config(cfg)
    normal = model.half_spread_per_oz(NORMAL_HOUR)
    thin = model.half_spread_per_oz(ASIAN_HOUR)
    assert thin > normal
    # 2.5x multiplier on the spread only; slippage is unchanged.
    assert thin == pytest.approx(0.15 * 2.5)
    assert model.cost_per_oz(ASIAN_HOUR) == pytest.approx(0.375 + 0.10)


def test_news_window_widens_the_spread_further(flat_cost_model):
    event = pd.Timestamp("2024-01-10 13:30", tz="UTC")
    model = CostModel(
        spread_usd_per_oz_round_trip=0.30, slippage_usd_per_oz_per_side=0.10,
        news_spread_multiplier=3.0, news_window_minutes=15,
        news_timestamps=(event,),
    )
    assert model.half_spread_per_oz(event) == pytest.approx(0.45)
    assert model.half_spread_per_oz(event + pd.Timedelta(minutes=10)) == pytest.approx(0.45)
    assert model.half_spread_per_oz(event + pd.Timedelta(hours=2)) == pytest.approx(0.15)


def test_spread_multiplier_series_matches_the_scalar_lookup(cfg):
    model = CostModel.from_config(cfg)
    index = pd.date_range("2024-01-03", periods=48, freq="1h", tz="UTC")
    series = model.spread_multiplier_series(index)
    for ts in index[::7]:
        assert series[ts] == pytest.approx(model.spread_multiplier(ts))


def test_commission_is_charged_per_lot_not_per_ounce():
    model = CostModel(
        spread_usd_per_oz_round_trip=0.0, slippage_usd_per_oz_per_side=0.0,
        commission_usd_per_lot_per_side=5.0, contract_size_oz_per_lot=100.0,
    )
    # 100 oz is exactly one lot.
    assert model.trade_costs(100.0, BUY, NORMAL_HOUR).commission == pytest.approx(5.0)
    assert model.trade_costs(1.0, BUY, NORMAL_HOUR).commission == pytest.approx(0.05)


def test_costs_scale_linearly_with_size(flat_cost_model):
    one = flat_cost_model.trade_costs(1.0, BUY, NORMAL_HOUR).total
    ten = flat_cost_model.trade_costs(10.0, BUY, NORMAL_HOUR).total
    assert ten == pytest.approx(10 * one)


def test_zero_size_costs_nothing(flat_cost_model):
    assert flat_cost_model.trade_costs(0.0, BUY, NORMAL_HOUR).total == 0.0


def test_side_must_be_plus_or_minus_one(flat_cost_model):
    with pytest.raises(ValueError):
        flat_cost_model.effective_fill_price(1800.0, 0, NORMAL_HOUR)


# ---------------------------------------------------------------------- #
# Fills always move against the trade
# ---------------------------------------------------------------------- #
def test_effective_fill_prices_are_adverse(flat_cost_model):
    assert flat_cost_model.effective_fill_price(1800.0, BUY, NORMAL_HOUR) == pytest.approx(1800.25)
    assert flat_cost_model.effective_fill_price(1800.0, SELL, NORMAL_HOUR) == pytest.approx(1799.75)


# ---------------------------------------------------------------------- #
# Financing
# ---------------------------------------------------------------------- #
def test_long_financing_is_a_charge(flat_cost_model):
    # swap_long is -0.15 per oz per night, i.e. you pay. Three nights on 1 oz.
    assert flat_cost_model.financing_cost(1.0, 3) == pytest.approx(0.45)


def test_short_financing_is_a_credit(flat_cost_model):
    # swap_short is +0.05, i.e. you receive, so the cost is negative.
    assert flat_cost_model.financing_cost(-1.0, 2) == pytest.approx(-0.10)


def test_no_financing_when_flat_or_when_no_night_passes(flat_cost_model):
    assert flat_cost_model.financing_cost(0.0, 5) == 0.0
    assert flat_cost_model.financing_cost(1.0, 0) == 0.0


# ---------------------------------------------------------------------- #
# TradeCosts arithmetic
# ---------------------------------------------------------------------- #
def test_trade_costs_add_componentwise():
    a = TradeCosts(spread=1.0, slippage=2.0, commission=3.0, financing=4.0)
    b = TradeCosts(spread=0.5, slippage=0.5, commission=0.5, financing=0.5)
    total = a + b
    assert (total.spread, total.slippage, total.commission, total.financing) == (1.5, 2.5, 3.5, 4.5)
    assert total.total == pytest.approx(12.0)


# ---------------------------------------------------------------------- #
# Portfolio
# ---------------------------------------------------------------------- #
def test_hand_computed_long_round_trip(flat_cost_model):
    """Buy 1 oz at 1800, sell at 1810.

    gross          = 1 * (1810 - 1800)      = 10.00
    costs          = 2 sides * 1 oz * 0.25  =  0.50
    net            = 10.00 - 0.50           =  9.50
    effective fills: buy 1800.25, sell 1809.75 -> 9.50, the same number twice.
    """
    portfolio = Portfolio(1000.0, flat_cost_model)
    portfolio.open_position(ts=NORMAL_HOUR, bar_index=0, direction=1, size_oz=1.0,
                            raw_price=1800.0, stop_distance_usd_per_oz=5.0, risk_usd=5.0)
    trade = portfolio.close_position(ts=NORMAL_HOUR + pd.Timedelta(hours=1),
                                     bar_index=4, raw_price=1810.0)

    assert trade.gross_pnl == pytest.approx(10.00)
    assert trade.total_cost == pytest.approx(0.50)
    assert trade.net_pnl == pytest.approx(9.50)
    assert trade.entry_price_eff == pytest.approx(1800.25)
    assert trade.exit_price_eff == pytest.approx(1809.75)
    assert trade.size_oz * (trade.exit_price_eff - trade.entry_price_eff) == pytest.approx(9.50)
    # R multiple: 9.50 net against 5.00 of risk.
    assert trade.r_multiple == pytest.approx(1.9)
    assert trade.bars_held == 4
    assert portfolio.net_equity == pytest.approx(1009.50)
    assert portfolio.gross_equity == pytest.approx(1010.00)


def test_hand_computed_short_round_trip(flat_cost_model):
    """Sell 2 oz at 1810, buy back at 1800.

    gross = -2 * (1800 - 1810) = 20.00 ; costs = 2 sides * 2 oz * 0.25 = 1.00
    """
    portfolio = Portfolio(1000.0, flat_cost_model)
    portfolio.open_position(ts=NORMAL_HOUR, bar_index=0, direction=-1, size_oz=2.0,
                            raw_price=1810.0)
    trade = portfolio.close_position(ts=NORMAL_HOUR, bar_index=2, raw_price=1800.0)
    assert trade.gross_pnl == pytest.approx(20.00)
    assert trade.total_cost == pytest.approx(1.00)
    assert trade.net_pnl == pytest.approx(19.00)
    assert portfolio.net_equity == pytest.approx(1019.00)


def test_losing_trade_costs_more_than_the_price_move(flat_cost_model):
    """A one-ounce trade that goes 0.20 the right way still loses money.

    This is the arithmetic that decides the whole project: the round trip costs
    0.50 per ounce, so anything smaller than that is a loss dressed as a win.
    """
    portfolio = Portfolio(50.0, flat_cost_model)
    portfolio.open_position(ts=NORMAL_HOUR, bar_index=0, direction=1, size_oz=1.0,
                            raw_price=1800.0)
    trade = portfolio.close_position(ts=NORMAL_HOUR, bar_index=1, raw_price=1800.20)
    assert trade.gross_pnl == pytest.approx(0.20)
    assert trade.net_pnl == pytest.approx(-0.30)


def test_financing_is_attributed_to_the_trade_that_incurred_it(flat_cost_model):
    portfolio = Portfolio(1000.0, flat_cost_model)
    portfolio.open_position(ts=NORMAL_HOUR, bar_index=0, direction=1, size_oz=1.0,
                            raw_price=1800.0)
    portfolio.accrue_financing(2)
    trade = portfolio.close_position(ts=NORMAL_HOUR, bar_index=10, raw_price=1800.0)
    assert trade.financing_cost == pytest.approx(0.30)
    assert trade.total_cost == pytest.approx(0.50 + 0.30)
    assert trade.net_pnl == pytest.approx(-0.80)


def test_unrealised_pnl_is_marked_but_costs_are_already_paid(flat_cost_model):
    portfolio = Portfolio(1000.0, flat_cost_model)
    portfolio.open_position(ts=NORMAL_HOUR, bar_index=0, direction=1, size_oz=1.0,
                            raw_price=1800.0)
    # Entry costs hit net equity immediately, which is what a real account sees.
    assert portfolio.gross_equity == pytest.approx(1000.0)
    assert portfolio.net_equity == pytest.approx(999.75)
    portfolio.mark(1805.0, 3)
    assert portfolio.unrealised_gross == pytest.approx(5.0)
    assert portfolio.net_equity == pytest.approx(1004.75)


def test_reconciliation_agrees_by_two_independent_routes(flat_cost_model):
    """Net from effective fill prices must equal gross minus costs."""
    rng = np.random.default_rng(0)
    portfolio = Portfolio(10_000.0, flat_cost_model)
    price = 1800.0
    for i in range(30):
        direction = int(rng.choice([-1, 1]))
        portfolio.open_position(ts=NORMAL_HOUR, bar_index=2 * i, direction=direction,
                                size_oz=float(rng.choice([1.0, 2.0, 5.0])), raw_price=price)
        portfolio.accrue_financing(int(rng.integers(0, 3)))
        price += float(rng.normal(0, 4))
        portfolio.close_position(ts=NORMAL_HOUR, bar_index=2 * i + 1, raw_price=price)

    reconciliation = portfolio.reconciliation()
    assert reconciliation["difference"] == pytest.approx(0.0, abs=1e-9)


def test_cannot_open_two_positions_at_once(flat_cost_model):
    portfolio = Portfolio(1000.0, flat_cost_model)
    portfolio.open_position(ts=NORMAL_HOUR, bar_index=0, direction=1, size_oz=1.0, raw_price=1800.0)
    with pytest.raises(RuntimeError):
        portfolio.open_position(ts=NORMAL_HOUR, bar_index=1, direction=1, size_oz=1.0, raw_price=1801.0)


def test_closing_when_flat_is_a_no_op(flat_cost_model):
    portfolio = Portfolio(1000.0, flat_cost_model)
    assert portfolio.close_position(ts=NORMAL_HOUR, bar_index=0, raw_price=1800.0) is None


def test_negative_or_zero_capital_is_rejected(flat_cost_model):
    with pytest.raises(ValueError):
        Portfolio(0.0, flat_cost_model)
