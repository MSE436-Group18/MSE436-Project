from __future__ import annotations

from dataclasses import replace

import pytest

from src.decision import (
    InvestmentAssumptions,
    evaluate_investment,
    monthly_mortgage_payment,
)
from src.model import ValuationRange


def base_assumptions() -> InvestmentAssumptions:
    return InvestmentAssumptions(
        asking_price=345000.0,
        monthly_rent=3100.0,
        down_payment_pct=25.0,
        annual_interest_rate_pct=5.1,
        amortization_years=25,
        vacancy_pct=4.0,
        annual_property_tax_pct=1.2,
        annual_insurance=1500.0,
        maintenance_pct_of_rent=8.0,
        annual_rent_growth_pct=3.0,
        annual_appreciation_pct=3.5,
        holding_period_years=7,
        closing_cost_pct=2.5,
        sale_cost_pct=5.0,
        target_annual_return_pct=9.0,
        risk_profile="Balanced",
    )


def test_zero_interest_mortgage_is_evenly_amortized() -> None:
    payment = monthly_mortgage_payment(
        principal=300000.0,
        annual_interest_rate_pct=0.0,
        amortization_years=25,
    )
    assert payment == pytest.approx(1000.0)


def test_rate_shock_reduces_cash_flow_and_return() -> None:
    valuation = ValuationRange(lower=199000.0, expected=220000.0, upper=251000.0)
    base = evaluate_investment(
        valuation=valuation,
        valuation_basis="Expected",
        market_multiplier=1.60,
        assumptions=base_assumptions(),
    )
    shocked = evaluate_investment(
        valuation=valuation,
        valuation_basis="Conservative",
        market_multiplier=1.60,
        assumptions=replace(
            base_assumptions(),
            asking_price=395000.0,
            monthly_rent=2850.0,
            down_payment_pct=20.0,
            annual_interest_rate_pct=8.4,
            vacancy_pct=10.0,
            annual_appreciation_pct=1.0,
            annual_rent_growth_pct=1.0,
            maintenance_pct_of_rent=12.0,
            target_annual_return_pct=10.0,
            risk_profile="Conservative",
        ),
    )

    assert shocked.monthly_cash_flow < base.monthly_cash_flow
    assert shocked.annualized_return_pct < base.annualized_return_pct
    assert base.recommendation == "BUY"
    assert shocked.recommendation == "AVOID"


def test_valuation_basis_is_load_bearing() -> None:
    valuation = ValuationRange(lower=190000.0, expected=220000.0, upper=260000.0)
    conservative = evaluate_investment(
        valuation=valuation,
        valuation_basis="Conservative",
        market_multiplier=1.60,
        assumptions=base_assumptions(),
    )
    optimistic = evaluate_investment(
        valuation=valuation,
        valuation_basis="Optimistic",
        market_multiplier=1.60,
        assumptions=base_assumptions(),
    )

    assert conservative.selected_fair_value < optimistic.selected_fair_value
    assert conservative.valuation_margin_pct < optimistic.valuation_margin_pct
    assert conservative.recommendation != optimistic.recommendation
