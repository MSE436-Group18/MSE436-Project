from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.decision import InvestmentAssumptions, evaluate_investment
from src.model import PropertyFeatures, ValuationModelBundle, load_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "valuation_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "model_metrics.json"

PRESETS: dict[str, dict[str, Any]] = {
    "Value Play": {
        "overall_quality": 7,
        "overall_condition": 6,
        "living_area_sqft": 1850,
        "bedrooms": 3,
        "full_bathrooms": 2,
        "year_built": 2003,
        "garage_capacity": 2.0,
        "basement_sqft": 980.0,
        "neighborhood": "Somerst",
        "building_type": "1Fam",
        "kitchen_quality": "Gd",
        "asking_price": 345000.0,
        "monthly_rent": 3100.0,
        "market_multiplier": 1.60,
        "valuation_basis": "Expected",
        "risk_profile": "Balanced",
        "down_payment_pct": 25.0,
        "interest_rate_pct": 5.1,
        "vacancy_pct": 4.0,
        "appreciation_pct": 3.5,
        "rent_growth_pct": 3.0,
        "target_return_pct": 9.0,
        "maintenance_pct": 8.0,
        "property_tax_pct": 1.2,
        "annual_insurance": 1500.0,
        "holding_period_years": 7,
    },
    "Rate Shock": {
        "overall_quality": 7,
        "overall_condition": 6,
        "living_area_sqft": 1850,
        "bedrooms": 3,
        "full_bathrooms": 2,
        "year_built": 2003,
        "garage_capacity": 2.0,
        "basement_sqft": 980.0,
        "neighborhood": "Somerst",
        "building_type": "1Fam",
        "kitchen_quality": "Gd",
        "asking_price": 395000.0,
        "monthly_rent": 2850.0,
        "market_multiplier": 1.60,
        "valuation_basis": "Conservative",
        "risk_profile": "Conservative",
        "down_payment_pct": 20.0,
        "interest_rate_pct": 8.4,
        "vacancy_pct": 10.0,
        "appreciation_pct": 1.0,
        "rent_growth_pct": 1.0,
        "target_return_pct": 10.0,
        "maintenance_pct": 12.0,
        "property_tax_pct": 1.2,
        "annual_insurance": 1500.0,
        "holding_period_years": 7,
    },
    "Premium Listing": {
        "overall_quality": 5,
        "overall_condition": 5,
        "living_area_sqft": 1260,
        "bedrooms": 3,
        "full_bathrooms": 1,
        "year_built": 1948,
        "garage_capacity": 1.0,
        "basement_sqft": 620.0,
        "neighborhood": "OldTown",
        "building_type": "1Fam",
        "kitchen_quality": "TA",
        "asking_price": 275000.0,
        "monthly_rent": 1750.0,
        "market_multiplier": 1.60,
        "valuation_basis": "Expected",
        "risk_profile": "Balanced",
        "down_payment_pct": 20.0,
        "interest_rate_pct": 6.2,
        "vacancy_pct": 7.0,
        "appreciation_pct": 2.0,
        "rent_growth_pct": 2.0,
        "target_return_pct": 9.0,
        "maintenance_pct": 12.0,
        "property_tax_pct": 1.3,
        "annual_insurance": 1450.0,
        "holding_period_years": 7,
    },
}


@st.cache_resource
def get_model() -> ValuationModelBundle:
    return load_model(MODEL_PATH)


@st.cache_data
def get_metrics() -> dict[str, Any]:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


def initialize_state() -> None:
    if "dss_initialized" in st.session_state:
        return
    requested_preset = st.query_params.get("preset", "Value Play")
    if requested_preset not in PRESETS:
        requested_preset = "Value Play"
    st.session_state["scenario_preset"] = requested_preset
    apply_preset(requested_preset)
    st.session_state["dss_initialized"] = True


def on_preset_change() -> None:
    apply_preset(st.session_state["scenario_preset"])


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def valuation_chart(
    lower: float,
    selected: float,
    upper: float,
    asking_price: float,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            y=["Valuation"],
            x=[upper - lower],
            base=[lower],
            orientation="h",
            marker={"color": "#9db6a6", "line": {"width": 0}},
            name="80% model range",
            hovertemplate=f"80% range: {money(lower)} to {money(upper)}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[selected],
            y=["Valuation"],
            mode="markers+text",
            marker={"color": "#17211b", "size": 14, "symbol": "diamond"},
            text=[f"Selected {money(selected)}"],
            textposition="top center",
            name="Selected model value",
            hovertemplate=f"Selected model value: {money(selected)}<extra></extra>",
        )
    )
    figure.add_vline(
        x=asking_price,
        line={"color": "#d85b2f", "width": 3, "dash": "dash"},
        annotation_text=f"Asking {money(asking_price)}",
        annotation_position="bottom right",
    )
    minimum = max(0.0, min(lower, asking_price) * 0.82)
    maximum = max(upper, asking_price) * 1.12
    figure.update_layout(
        height=280,
        margin={"l": 6, "r": 12, "t": 42, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis={
            "range": [minimum, maximum],
            "tickprefix": "$",
            "tickformat": ",.0f",
            "gridcolor": "rgba(23,33,27,0.10)",
            "title": None,
        },
        yaxis={"showticklabels": False, "title": None},
        font={"family": "DM Sans, sans-serif", "color": "#17211b"},
    )
    return figure


def scenario_chart(
    scenario_names: list[str],
    returns: list[float],
    target: float,
) -> go.Figure:
    colors = ["#b94d3f", "#d85b2f", "#3f7255"]
    figure = go.Figure(
        go.Bar(
            x=scenario_names,
            y=returns,
            marker={"color": colors},
            text=[f"{value:.1f}%" for value in returns],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}% annualized<extra></extra>",
        )
    )
    figure.add_hline(
        y=target,
        line={"color": "#17211b", "width": 2, "dash": "dot"},
        annotation_text=f"Decision target {target:.1f}%",
        annotation_position="top left",
    )
    lower_bound = min(0.0, min(returns) - 3.0)
    upper_bound = max(target, max(returns)) + 5.0
    figure.update_layout(
        height=280,
        margin={"l": 6, "r": 12, "t": 42, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis={
            "range": [lower_bound, upper_bound],
            "ticksuffix": "%",
            "gridcolor": "rgba(23,33,27,0.10)",
            "title": None,
        },
        xaxis={"title": None},
        font={"family": "DM Sans, sans-serif", "color": "#17211b"},
    )
    return figure


def equity_chart(years: list[int], equity: list[float], cash_flow: list[float]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=years,
            y=equity,
            mode="lines+markers",
            line={"color": "#3f7255", "width": 4},
            marker={"size": 8},
            name="Investor equity",
            hovertemplate="Year %{x}: $%{y:,.0f} equity<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=years,
            y=cash_flow,
            marker={"color": "rgba(216,91,47,0.36)"},
            name="Cumulative cash flow",
            hovertemplate="Year %{x}: $%{y:,.0f} cumulative cash flow<extra></extra>",
        )
    )
    figure.update_layout(
        height=300,
        margin={"l": 6, "r": 12, "t": 24, "b": 22},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        yaxis={
            "tickprefix": "$",
            "tickformat": ",.0f",
            "gridcolor": "rgba(23,33,27,0.10)",
            "title": None,
        },
        xaxis={"dtick": 1, "title": "Holding year"},
        font={"family": "DM Sans, sans-serif", "color": "#17211b"},
    )
    return figure


st.set_page_config(
    page_title="Brickwise Property Decision Lab",
    page_icon="BW",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
    :root {
        --ink: #17211b;
        --paper: #f4efe3;
        --sage: #3f7255;
        --sage-soft: #dfe8dc;
        --orange: #d85b2f;
        --red: #a64035;
        --line: rgba(23, 33, 27, 0.14);
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    [data-testid="stAppViewContainer"] {
        color: var(--ink);
        background:
            radial-gradient(circle at 83% 4%, rgba(216,91,47,0.13), transparent 28rem),
            linear-gradient(rgba(23,33,27,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(23,33,27,0.025) 1px, transparent 1px),
            var(--paper);
        background-size: auto, 32px 32px, 32px 32px, auto;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none; }
    [data-testid="stSidebar"] {
        background: #e5eadf;
        border-right: 1px solid var(--line);
        min-width: 360px !important;
        width: 360px !important;
    }
    [data-testid="stSidebar"] > div:first-child { width: 360px !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { font-family: 'Fraunces', Georgia, serif; }
    .block-container { padding-top: 2.25rem; padding-bottom: 3rem; max-width: 1500px; }
    .hero-kicker {
        display: inline-block;
        color: var(--orange);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .hero-title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: clamp(2.35rem, 4.6vw, 4.65rem);
        line-height: 0.98;
        letter-spacing: -0.045em;
        max-width: 980px;
        margin: 0;
        color: var(--ink);
        overflow-wrap: anywhere;
    }
    .hero-subtitle {
        max-width: 810px;
        font-size: 1.05rem;
        line-height: 1.55;
        color: rgba(23,33,27,0.76);
        margin: 1rem 0 1.65rem;
    }
    .decision-banner {
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.15rem 1.35rem 1.25rem;
        box-shadow: 0 16px 38px rgba(23,33,27,0.09);
        margin-bottom: 1rem;
    }
    .decision-banner.buy { background: linear-gradient(135deg, #dcebdc, #edf1e5); }
    .decision-banner.watch { background: linear-gradient(135deg, #f4e5bd, #f3ecdc); }
    .decision-banner.avoid { background: linear-gradient(135deg, #f1d7cf, #f3e8df); }
    .decision-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700;
        opacity: 0.72;
    }
    .decision-word {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 2.55rem;
        line-height: 1;
        margin: 0.2rem 0 0.4rem;
    }
    .decision-rationale { font-size: 1rem; line-height: 1.48; max-width: 980px; }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.8rem 0 1.35rem;
    }
    .metric-card {
        min-height: 112px;
        padding: 0.95rem 1rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255,255,255,0.54);
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.62;
    }
    .metric-value {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.55rem;
        margin-top: 0.3rem;
    }
    .metric-note { font-size: 0.78rem; opacity: 0.65; margin-top: 0.18rem; }
    .evidence-card {
        border-top: 3px solid var(--sage);
        padding: 0.9rem 1rem;
        background: rgba(255,255,255,0.46);
        min-height: 170px;
    }
    .evidence-card.risk { border-color: var(--orange); }
    .evidence-card h4 {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.2rem;
        margin: 0 0 0.55rem;
    }
    .evidence-card ul { padding-left: 1.1rem; margin: 0; }
    .evidence-card li { margin: 0.38rem 0; line-height: 1.4; }
    .section-title {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.55rem;
        margin: 1.5rem 0 0.15rem;
    }
    .section-note { opacity: 0.66; margin-bottom: 0.6rem; }
    @media (max-width: 900px) {
        .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .hero-title { font-size: 2.65rem; }
    }
    @media (max-width: 560px) {
        .metric-grid { grid-template-columns: 1fr; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error(
        "Model artifacts are missing. Run `python scripts/train_model.py` from the project root."
    )
    st.stop()

model = get_model()
metrics = get_metrics()
initialize_state()

with st.sidebar:
    st.title("Scenario controls")
    st.caption("Change any orange-labeled assumption and the investment decision recalculates.")
    st.selectbox(
        "Demo preset",
        options=list(PRESETS),
        key="scenario_preset",
        on_change=on_preset_change,
        help="Loads a complete, reproducible decision scenario.",
    )
    st.markdown("#### Decision levers")
    st.number_input(
        "Asking price ($)",
        min_value=50000.0,
        max_value=2000000.0,
        step=5000.0,
        key="asking_price",
    )
    st.number_input(
        "Expected monthly rent ($)",
        min_value=500.0,
        max_value=15000.0,
        step=50.0,
        key="monthly_rent",
    )
    st.slider(
        "Mortgage rate (%)",
        min_value=0.0,
        max_value=12.0,
        step=0.1,
        key="interest_rate_pct",
    )
    st.slider(
        "Vacancy (%)",
        min_value=0.0,
        max_value=25.0,
        step=0.5,
        key="vacancy_pct",
    )
    st.slider(
        "Expected appreciation (%)",
        min_value=-5.0,
        max_value=10.0,
        step=0.25,
        key="appreciation_pct",
    )
    st.slider(
        "Minimum annual return (%)",
        min_value=0.0,
        max_value=25.0,
        step=0.5,
        key="target_return_pct",
    )
    st.selectbox(
        "Valuation basis",
        options=["Conservative", "Expected", "Optimistic"],
        key="valuation_basis",
        help="Selects the lower, expected, or upper estimate from the model range.",
    )
    st.selectbox(
        "Risk policy",
        options=["Conservative", "Balanced", "Growth"],
        key="risk_profile",
    )

    with st.expander("Property facts used by the model"):
        st.slider("Overall quality", 1, 10, key="overall_quality")
        st.slider("Overall condition", 1, 10, key="overall_condition")
        st.number_input("Living area (sq ft)", 400, 6000, step=50, key="living_area_sqft")
        st.number_input("Bedrooms", 0, 8, step=1, key="bedrooms")
        st.number_input("Full bathrooms", 0, 6, step=1, key="full_bathrooms")
        st.number_input("Year built", 1870, 2026, step=1, key="year_built")
        st.number_input("Garage capacity", 0.0, 5.0, step=1.0, key="garage_capacity")
        st.number_input("Basement area (sq ft)", 0.0, 4000.0, step=50.0, key="basement_sqft")
        st.selectbox("Neighborhood", model.neighborhoods, key="neighborhood")
        st.selectbox("Building type", model.building_types, key="building_type")
        st.selectbox("Kitchen quality", model.kitchen_qualities, key="kitchen_quality")
        st.slider(
            "Market index multiplier",
            0.5,
            3.0,
            step=0.05,
            key="market_multiplier",
            help="Updates historical Ames-dollar model output to the user's market index.",
        )

    with st.expander("Financing and operating assumptions"):
        st.slider("Down payment (%)", 5.0, 100.0, step=1.0, key="down_payment_pct")
        st.slider("Annual rent growth (%)", -5.0, 10.0, step=0.25, key="rent_growth_pct")
        st.slider("Maintenance (% of rent)", 0.0, 30.0, step=0.5, key="maintenance_pct")
        st.slider(
            "Property tax (% of value)",
            0.0,
            4.0,
            step=0.1,
            key="property_tax_pct",
        )
        st.number_input(
            "Annual insurance ($)",
            0.0,
            20000.0,
            step=100.0,
            key="annual_insurance",
        )
        st.slider("Holding period (years)", 2, 15, step=1, key="holding_period_years")

property_features = PropertyFeatures(
    overall_quality=st.session_state["overall_quality"],
    overall_condition=st.session_state["overall_condition"],
    living_area_sqft=st.session_state["living_area_sqft"],
    bedrooms=st.session_state["bedrooms"],
    full_bathrooms=st.session_state["full_bathrooms"],
    year_built=st.session_state["year_built"],
    garage_capacity=st.session_state["garage_capacity"],
    basement_sqft=st.session_state["basement_sqft"],
    neighborhood=st.session_state["neighborhood"],
    building_type=st.session_state["building_type"],
    kitchen_quality=st.session_state["kitchen_quality"],
)
assumptions = InvestmentAssumptions(
    asking_price=st.session_state["asking_price"],
    monthly_rent=st.session_state["monthly_rent"],
    down_payment_pct=st.session_state["down_payment_pct"],
    annual_interest_rate_pct=st.session_state["interest_rate_pct"],
    amortization_years=25,
    vacancy_pct=st.session_state["vacancy_pct"],
    annual_property_tax_pct=st.session_state["property_tax_pct"],
    annual_insurance=st.session_state["annual_insurance"],
    maintenance_pct_of_rent=st.session_state["maintenance_pct"],
    annual_rent_growth_pct=st.session_state["rent_growth_pct"],
    annual_appreciation_pct=st.session_state["appreciation_pct"],
    holding_period_years=st.session_state["holding_period_years"],
    closing_cost_pct=2.5,
    sale_cost_pct=5.0,
    target_annual_return_pct=st.session_state["target_return_pct"],
    risk_profile=st.session_state["risk_profile"],
)

valuation = model.predict(property_features)
decision = evaluate_investment(
    valuation=valuation,
    valuation_basis=st.session_state["valuation_basis"],
    market_multiplier=st.session_state["market_multiplier"],
    assumptions=assumptions,
)

st.markdown(
    """
    <div class="hero-kicker">Human-in-the-loop investment DSS</div>
    <h1 class="hero-title">Brickwise Property<br>Decision Lab</h1>
    <p class="hero-subtitle">
      A trained valuation model estimates a defensible price range. Your financing,
      operating, and risk assumptions then determine whether this property clears the
      investment policy as a BUY, WATCH, or AVOID.
    </p>
    """,
    unsafe_allow_html=True,
)

banner_class = decision.recommendation.lower()
st.markdown(
    f"""
    <div class="decision-banner {banner_class}">
      <div class="decision-label">Current recommendation | decision score {decision.score}/100</div>
      <div class="decision-word">{decision.recommendation}</div>
      <div class="decision-rationale">{decision.rationale}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Selected fair value</div>
        <div class="metric-value">{money(decision.selected_fair_value)}</div>
        <div class="metric-note">{st.session_state["valuation_basis"]} model basis</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Valuation margin</div>
        <div class="metric-value">{decision.valuation_margin_pct:+.1f}%</div>
        <div class="metric-note">relative to asking price</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Year-one cash flow</div>
        <div class="metric-value">{money(decision.monthly_cash_flow)}/mo</div>
        <div class="metric-note">after debt and operating costs</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Annualized return</div>
        <div class="metric-value">{decision.annualized_return_pct:.1f}%</div>
        <div class="metric-note">target {decision.target_return_pct:.1f}%</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

chart_left, chart_right = st.columns(2, gap="large")
with chart_left:
    st.markdown('<div class="section-title">Price decision</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">The asking price must make sense inside the model range.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        valuation_chart(
            lower=decision.adjusted_valuation.lower,
            selected=decision.selected_fair_value,
            upper=decision.adjusted_valuation.upper,
            asking_price=assumptions.asking_price,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with chart_right:
    st.markdown('<div class="section-title">Return decision</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">The base and downside cases are tested '
        "against the target.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        scenario_chart(
            scenario_names=[scenario.name for scenario in decision.scenarios],
            returns=[scenario.annualized_return_pct for scenario in decision.scenarios],
            target=decision.target_return_pct,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

st.markdown('<div class="section-title">Decision evidence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Every item below maps directly to a BUY / WATCH / AVOID rule.</div>',
    unsafe_allow_html=True,
)
evidence_left, evidence_right = st.columns(2, gap="large")
with evidence_left:
    driver_items = "".join(f"<li>{item}</li>" for item in decision.drivers)
    st.markdown(
        f'<div class="evidence-card"><h4>What supports the deal</h4><ul>{driver_items}</ul></div>',
        unsafe_allow_html=True,
    )
with evidence_right:
    risk_items = "".join(f"<li>{item}</li>" for item in decision.risks)
    if not risk_items:
        risk_items = "<li>No policy-breaking risk was found in the selected scenario.</li>"
    st.markdown(
        f'<div class="evidence-card risk"><h4>What could stop the deal</h4>'
        f"<ul>{risk_items}</ul></div>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Holding-period outcome</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Equity growth and cumulative cash flow show '
    "how the return is created.</div>",
    unsafe_allow_html=True,
)
st.plotly_chart(
    equity_chart(
        years=[year.year for year in decision.projection],
        equity=[year.investor_equity for year in decision.projection],
        cash_flow=[year.cumulative_cash_flow for year in decision.projection],
    ),
    use_container_width=True,
    config={"displayModeBar": False},
)

with st.expander("Model evidence and limitations"):
    st.markdown(
        f"""
        - **Task:** supervised regression of residential sale price, with lower and upper
          quantile models providing an 80% valuation range.
        - **Temporal validation:** {metrics["training_rows"]:,} sales from
          {metrics["training_years"][0]}-{metrics["training_years"][1]} trained the model;
          {metrics["test_rows"]:,} sales from {metrics["test_year"]} were held out.
        - **Holdout performance:** MAE {money(metrics["mae"])}, R-squared
          {metrics["r2"]:.3f}, and {metrics["interval_coverage_pct"]:.1f}% interval coverage.
        - **Why this model:** gradient-boosted trees capture non-linear price effects and
          interactions while remaining fast enough for live what-if analysis.
        - **Limits:** Ames is one U.S. city and the sales are historical. The market-index
          multiplier is explicit, but a deployed system must retrain on current local sales,
          rents, vacancy, and interest-rate data before any real investment decision.
        """
    )

st.caption(
    "Course prototype only. Outputs are scenario estimates, not financial advice. "
    "Ames Housing source: De Cock (2011), Journal of Statistics Education."
)
