from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RiskProfile = Literal["Conservative", "Balanced", "Growth"]


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
