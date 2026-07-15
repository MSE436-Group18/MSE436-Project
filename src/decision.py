from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from src.model import ValuationBasis, ValuationRange

RiskProfile = Literal["Conservative", "Balanced", "Growth"]
Recommendation = Literal["BUY", "WATCH", "AVOID"]


@dataclass(frozen=True)
class InvestmentAssumptions:
    asking_price: float
    monthly_rent: float
    down_payment_pct: float
    annual_interest_rate_pct: float
    amortization_years: int
    vacancy_pct: float
    annual_property_tax_pct: float
    annual_insurance: float
    maintenance_pct_of_rent: float
    annual_rent_growth_pct: float
    annual_appreciation_pct: float
    holding_period_years: int
    closing_cost_pct: float
    sale_cost_pct: float
    target_annual_return_pct: float
    risk_profile: RiskProfile


@dataclass(frozen=True)
class YearProjection:
    year: int
    property_value: float
    loan_balance: float
    investor_equity: float
    annual_cash_flow: float
    cumulative_cash_flow: float


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    annualized_return_pct: float
    year_one_monthly_cash_flow: float
    cash_on_cash_return_pct: float
    total_profit: float
    projection: tuple[YearProjection, ...]


@dataclass(frozen=True)
class DecisionResult:
    recommendation: Recommendation
    score: int
    selected_fair_value: float
    adjusted_valuation: ValuationRange
    valuation_margin_pct: float
    monthly_cash_flow: float
    cash_on_cash_return_pct: float
    annualized_return_pct: float
    downside_return_pct: float
    target_return_pct: float
    scenarios: tuple[ScenarioResult, ...]
    projection: tuple[YearProjection, ...]
    drivers: tuple[str, ...]
    risks: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class _Policy:
    minimum_valuation_margin_pct: float
    downside_gap_from_target_pct: float
    minimum_monthly_cash_flow: float


POLICIES: dict[RiskProfile, _Policy] = {
    "Conservative": _Policy(
        minimum_valuation_margin_pct=3.0,
        downside_gap_from_target_pct=2.0,
        minimum_monthly_cash_flow=0.0,
    ),
    "Balanced": _Policy(
        minimum_valuation_margin_pct=-3.0,
        downside_gap_from_target_pct=5.0,
        minimum_monthly_cash_flow=-100.0,
    ),
    "Growth": _Policy(
        minimum_valuation_margin_pct=-8.0,
        downside_gap_from_target_pct=12.0,
        minimum_monthly_cash_flow=-300.0,
    ),
}


def monthly_mortgage_payment(
    principal: float,
    annual_interest_rate_pct: float,
    amortization_years: int,
) -> float:
    if principal < 0:
        raise ValueError("Principal cannot be negative.")
    if amortization_years <= 0:
        raise ValueError("Amortization period must be positive.")

    number_of_payments = amortization_years * 12
    monthly_rate = annual_interest_rate_pct / 100 / 12
    if monthly_rate == 0:
        return principal / number_of_payments
    growth = (1 + monthly_rate) ** number_of_payments
    return principal * monthly_rate * growth / (growth - 1)


def remaining_loan_balance(
    principal: float,
    annual_interest_rate_pct: float,
    amortization_years: int,
    payments_made: int,
) -> float:
    total_payments = amortization_years * 12
    completed_payments = min(max(payments_made, 0), total_payments)
    monthly_rate = annual_interest_rate_pct / 100 / 12
    if monthly_rate == 0:
        return principal * (1 - completed_payments / total_payments)

    total_growth = (1 + monthly_rate) ** total_payments
    completed_growth = (1 + monthly_rate) ** completed_payments
    return principal * (total_growth - completed_growth) / (total_growth - 1)


def _annualized_return(cash_flows: list[float]) -> float:
    def net_present_value(rate: float) -> float:
        return sum(
            cash_flow / ((1 + rate) ** period) for period, cash_flow in enumerate(cash_flows)
        )

    lower_rate = -0.95
    upper_rate = 10.0
    lower_value = net_present_value(lower_rate)
    upper_value = net_present_value(upper_rate)
    if lower_value * upper_value > 0:
        invested = abs(cash_flows[0])
        years = max(1, len(cash_flows) - 1)
        terminal_value = max(0.0, sum(cash_flows[1:]))
        if invested == 0 or terminal_value == 0:
            return -100.0
        return ((terminal_value / invested) ** (1 / years) - 1) * 100

    for _ in range(120):
        midpoint = (lower_rate + upper_rate) / 2
        midpoint_value = net_present_value(midpoint)
        if abs(midpoint_value) < 0.01:
            return midpoint * 100
        if lower_value * midpoint_value <= 0:
            upper_rate = midpoint
        else:
            lower_rate = midpoint
            lower_value = midpoint_value
    return ((lower_rate + upper_rate) / 2) * 100


def project_scenario(name: str, assumptions: InvestmentAssumptions) -> ScenarioResult:
    if assumptions.asking_price <= 0 or assumptions.monthly_rent < 0:
        raise ValueError("Asking price must be positive and rent cannot be negative.")

    down_payment = assumptions.asking_price * assumptions.down_payment_pct / 100
    closing_cost = assumptions.asking_price * assumptions.closing_cost_pct / 100
    initial_cash = down_payment + closing_cost
    principal = assumptions.asking_price - down_payment
    monthly_payment = monthly_mortgage_payment(
        principal,
        assumptions.annual_interest_rate_pct,
        assumptions.amortization_years,
    )

    cash_flows = [-initial_cash]
    projection: list[YearProjection] = []
    cumulative_cash_flow = 0.0

    for year in range(1, assumptions.holding_period_years + 1):
        rent_growth = (1 + assumptions.annual_rent_growth_pct / 100) ** (year - 1)
        gross_rent = assumptions.monthly_rent * 12 * rent_growth
        effective_rent = gross_rent * (1 - assumptions.vacancy_pct / 100)
        property_value = assumptions.asking_price * (
            (1 + assumptions.annual_appreciation_pct / 100) ** year
        )
        property_tax = property_value * assumptions.annual_property_tax_pct / 100
        maintenance = gross_rent * assumptions.maintenance_pct_of_rent / 100
        insurance = assumptions.annual_insurance * (1.02 ** (year - 1))
        annual_cash_flow = (
            effective_rent - property_tax - maintenance - insurance - monthly_payment * 12
        )
        cumulative_cash_flow += annual_cash_flow
        loan_balance = remaining_loan_balance(
            principal,
            assumptions.annual_interest_rate_pct,
            assumptions.amortization_years,
            year * 12,
        )
        investor_equity = property_value - loan_balance
        projection.append(
            YearProjection(
                year=year,
                property_value=property_value,
                loan_balance=loan_balance,
                investor_equity=investor_equity,
                annual_cash_flow=annual_cash_flow,
                cumulative_cash_flow=cumulative_cash_flow,
            )
        )
        cash_flows.append(annual_cash_flow)

    final_year = projection[-1]
    sale_proceeds = (
        final_year.property_value * (1 - assumptions.sale_cost_pct / 100) - final_year.loan_balance
    )
    cash_flows[-1] += sale_proceeds
    year_one_cash_flow = projection[0].annual_cash_flow
    cash_on_cash_return = year_one_cash_flow / initial_cash * 100

    return ScenarioResult(
        name=name,
        annualized_return_pct=_annualized_return(cash_flows),
        year_one_monthly_cash_flow=year_one_cash_flow / 12,
        cash_on_cash_return_pct=cash_on_cash_return,
        total_profit=sum(cash_flows),
        projection=tuple(projection),
    )


def _scenario_set(
    assumptions: InvestmentAssumptions,
) -> tuple[ScenarioResult, ScenarioResult, ScenarioResult]:
    downside = replace(
        assumptions,
        vacancy_pct=min(35.0, assumptions.vacancy_pct + 5.0),
        maintenance_pct_of_rent=min(35.0, assumptions.maintenance_pct_of_rent + 3.0),
        annual_rent_growth_pct=assumptions.annual_rent_growth_pct - 2.0,
        annual_appreciation_pct=assumptions.annual_appreciation_pct - 3.0,
    )
    upside = replace(
        assumptions,
        vacancy_pct=max(0.0, assumptions.vacancy_pct - 2.0),
        maintenance_pct_of_rent=max(0.0, assumptions.maintenance_pct_of_rent - 1.0),
        annual_rent_growth_pct=assumptions.annual_rent_growth_pct + 1.0,
        annual_appreciation_pct=assumptions.annual_appreciation_pct + 1.5,
    )
    return (
        project_scenario("Downside", downside),
        project_scenario("Base", assumptions),
        project_scenario("Upside", upside),
    )


def evaluate_investment(
    valuation: ValuationRange,
    valuation_basis: ValuationBasis,
    market_multiplier: float,
    assumptions: InvestmentAssumptions,
) -> DecisionResult:
    adjusted_valuation = valuation.adjusted(market_multiplier)
    selected_fair_value = adjusted_valuation.select(valuation_basis)
    valuation_margin_pct = (
        (selected_fair_value - assumptions.asking_price) / assumptions.asking_price * 100
    )
    downside, base, upside = _scenario_set(assumptions)
    policy = POLICIES[assumptions.risk_profile]
    downside_floor = assumptions.target_annual_return_pct - policy.downside_gap_from_target_pct

    qualifies_to_buy = all(
        [
            base.annualized_return_pct >= assumptions.target_annual_return_pct,
            downside.annualized_return_pct >= downside_floor,
            valuation_margin_pct >= policy.minimum_valuation_margin_pct,
            base.year_one_monthly_cash_flow >= policy.minimum_monthly_cash_flow,
        ]
    )
    clear_avoid = any(
        [
            base.annualized_return_pct < assumptions.target_annual_return_pct - 3.0,
            downside.annualized_return_pct < -2.0,
            valuation_margin_pct < policy.minimum_valuation_margin_pct - 12.0,
        ]
    )
    recommendation: Recommendation
    if qualifies_to_buy:
        recommendation = "BUY"
    elif clear_avoid:
        recommendation = "AVOID"
    else:
        recommendation = "WATCH"

    return_score = _clamp(
        50 + (base.annualized_return_pct - assumptions.target_annual_return_pct) * 4,
        0,
        60,
    )
    valuation_score = _clamp(15 + valuation_margin_pct, 0, 25)
    resilience_score = _clamp(10 + downside.annualized_return_pct, 0, 15)
    score = round(return_score + valuation_score + resilience_score)

    drivers: list[str] = []
    risks: list[str] = []
    if base.annualized_return_pct >= assumptions.target_annual_return_pct:
        drivers.append(
            f"Base return clears the target by "
            f"{base.annualized_return_pct - assumptions.target_annual_return_pct:.1f} pts."
        )
    else:
        risks.append(
            f"Base return misses the target by "
            f"{assumptions.target_annual_return_pct - base.annualized_return_pct:.1f} pts."
        )
    if valuation_margin_pct >= 0:
        drivers.append(
            f"Selected model value is {valuation_margin_pct:.1f}% above the asking price."
        )
    else:
        risks.append(
            f"Asking price is {-valuation_margin_pct:.1f}% above the selected model value."
        )
    if base.year_one_monthly_cash_flow >= 0:
        drivers.append(
            f"Year-one cash flow stays positive at ${base.year_one_monthly_cash_flow:,.0f}/month."
        )
    else:
        risks.append(
            f"Year-one cash flow requires "
            f"${-base.year_one_monthly_cash_flow:,.0f}/month of support."
        )
    if downside.annualized_return_pct >= downside_floor:
        drivers.append(
            f"Downside return of {downside.annualized_return_pct:.1f}% meets the "
            f"{assumptions.risk_profile.lower()} policy floor."
        )
    else:
        risks.append(
            f"Downside return of {downside.annualized_return_pct:.1f}% falls below the "
            f"{downside_floor:.1f}% policy floor."
        )

    rationale = (
        f"{recommendation}: projected annualized return is "
        f"{base.annualized_return_pct:.1f}% versus a "
        f"{assumptions.target_annual_return_pct:.1f}% target; the downside case is "
        f"{downside.annualized_return_pct:.1f}% and the selected valuation margin is "
        f"{valuation_margin_pct:+.1f}%."
    )
    return DecisionResult(
        recommendation=recommendation,
        score=score,
        selected_fair_value=selected_fair_value,
        adjusted_valuation=adjusted_valuation,
        valuation_margin_pct=valuation_margin_pct,
        monthly_cash_flow=base.year_one_monthly_cash_flow,
        cash_on_cash_return_pct=base.cash_on_cash_return_pct,
        annualized_return_pct=base.annualized_return_pct,
        downside_return_pct=downside.annualized_return_pct,
        target_return_pct=assumptions.target_annual_return_pct,
        scenarios=(downside, base, upside),
        projection=base.projection,
        drivers=tuple(drivers),
        risks=tuple(risks),
        rationale=rationale,
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
