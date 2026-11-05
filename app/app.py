from pathlib import Path
import io
import json
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "startuplens_complete_funding_data.xlsx"
METRICS = BASE / "model_comparison_metrics.csv"
REPORT = BASE / "evaluation_report.json"
MODELS = {
    "Logistic Regression": BASE / "startuplens_primary_logistic_regression.joblib",
    "Random Forest": BASE / "random_forest.joblib",
    "Extra Trees": BASE / "extra_trees.joblib",
}
TARGET_THRESHOLD = 0.50
TEAL = "#0E7C66"
INK = "#1A1714"
MUTED = "#6B645C"
RISK = "#B42318"
AMBER = "#C2410C"
NAVY = "#1E3A5F"
PAPER = "#F4F1EA"
CARD = "#FFFCF7"

FEATURE_LABELS = {
    "country": "Country",
    "city": "City",
    "sector_name": "Sector",
    "founded_year": "Founded year",
    "observed_funding_rounds": "Observed funding rounds",
    "dated_funding_rounds": "Rounds with known dates",
    "rounds_with_disclosed_amount": "Rounds with disclosed amount",
    "observed_total_funding_usd": "Total observed funding",
    "average_round_amount_usd": "Average round amount",
    "median_round_amount_usd": "Median round amount",
    "largest_round_amount_usd": "Largest round amount",
    "observed_funding_span_days": "Funding span (days)",
    "average_days_between_observed_rounds": "Average days between rounds",
    "median_days_between_observed_rounds": "Median days between rounds",
    "funding_gap_std_days": "Variation in funding gaps",
    "disclosed_amount_growth_first_to_last": "First-to-last round growth",
    "distinct_funding_stages": "Distinct funding stages",
    "latest_observed_stage": "Latest funding stage",
    "average_investor_count_per_round": "Average investors per round",
    "max_investor_count_in_round": "Maximum investors in a round",
    "total_news_mentions": "News mentions",
    "company_age_at_first_observed_round_years": "Age at first observed round",
    "company_age_at_last_observed_round_years": "Age at last observed round",
    "last_funding_gap_days": "Days since last round",
    "prior_funding_gap_median_days": "Prior median funding gap",
    "funding_slowdown_ratio": "Funding slowdown ratio",
    "funding_gap_change_days": "Change in funding gap",
    "recent_vs_previous_round_amount_growth": "Recent round amount growth",
    "recent_round_amount_share_of_total": "Largest round share of total",
    "amounts_disclosed_share": "Share of rounds with amounts",
    "investor_count_change_first_to_last": "Investor count change",
    "investor_count_growth_first_to_last": "Investor count growth",
    "investor_participation_trend": "Investor participation trend",
    "stage_advances_observed": "Observed stage advances",
    "highest_observed_stage_rank": "Highest observed stage rank",
    "distinct_known_stage_levels": "Distinct known stage levels",
    "rounds_per_observed_funding_year": "Rounds per funding year",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision_failure": "Failure precision",
    "recall_failure": "Failure recall",
    "f1_failure": "Failure F1",
    "roc_auc": "ROC-AUC",
    "pr_auc_average_precision": "PR-AUC",
}

st.set_page_config(page_title="StartupLens | Investor Intelligence", page_icon="SL", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,520;9..144,640&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: "Plus Jakarta Sans", sans-serif; }
.stApp { background: #F4F1EA; color: #1A1714; }
.block-container { padding-top: 1.1rem; padding-bottom: 2.4rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #FFFCF7; border-right: 1px solid #E7E1D6; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  font-family: Fraunces, serif; font-weight: 520;
}
h1, h2, h3 { font-family: Fraunces, serif !important; letter-spacing: -0.02em; color: #1A1714 !important; }
.stCaption, .small-note { color: #6B645C !important; }
[data-testid="stMetric"] {
  background: #FFFCF7; border: 1px solid #E7E1D6; padding: 16px 18px; border-radius: 16px;
  box-shadow: 0 1px 0 rgba(26,23,20,0.03);
}
[data-testid="stMetric"] label { color: #6B645C !important; font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-family: Fraunces, serif; font-weight: 520; }
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 8px; background: transparent; border-bottom: 1px solid #E7E1D6; padding-bottom: 8px;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent; color: #6B645C; border-radius: 999px; padding: 8px 16px;
  font-weight: 600;
}
div[data-testid="stTabs"] [aria-selected="true"] {
  background: #1A1714 !important; color: #FFFCF7 !important;
}
.sl-hero {
  display: flex; justify-content: space-between; gap: 24px; align-items: flex-end;
  padding: 28px 32px; border-radius: 24px; margin-bottom: 18px;
  background: linear-gradient(135deg, #FFFCF7 0%, #EFE8DC 55%, #DDECE6 100%);
  border: 1px solid #E7E1D6;
}
.sl-mark {
  width: 42px; height: 42px; border-radius: 14px; display: grid; place-items: center;
  background: #0E7C66; color: #F4F1EA; font-family: Fraunces, serif; font-weight: 640;
  font-size: 1.05rem; letter-spacing: -0.04em;
  animation: sl-float 4s ease-in-out infinite;
}
@keyframes sl-float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }
.sl-kicker { font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase; color: #0E7C66; font-weight: 700; margin: 0 0 6px; }
.sl-hero h1 { margin: 0; font-size: 2.15rem; line-height: 1.1; }
.sl-hero p { margin: 8px 0 0; color: #6B645C; max-width: 46rem; }
.sl-chip-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.sl-chip {
  background: #FFFCF7; border: 1px solid #E7E1D6; border-radius: 999px;
  padding: 7px 12px; font-size: 0.78rem; color: #3F3A36; font-weight: 600;
  transition: transform 160ms ease, box-shadow 160ms ease;
}
.sl-chip:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(14,124,102,0.12); }
.sl-callout {
  background: #FFFCF7; border: 1px solid #E7E1D6; border-radius: 18px; padding: 16px 18px; margin: 8px 0 16px;
}
.sl-flag-risk, .sl-flag-ok {
  border-radius: 18px; padding: 16px 18px; margin: 8px 0 16px; border: 1px solid;
}
.sl-flag-risk { background: #FDECEC; border-color: #F0B6B2; }
.sl-flag-ok { background: #E7F4F0; border-color: #B7D9D0; }
.sl-flag-risk h4, .sl-flag-ok h4 { margin: 0 0 4px; font-family: Fraunces, serif; }
.sl-flag-risk p, .sl-flag-ok p { margin: 0; color: #3F3A36; }
hr { border-color: #E7E1D6 !important; }
.stDownloadButton button, .stButton button[kind="primary"] {
  background: #0E7C66 !important; color: #fff !important; border: 0 !important; border-radius: 12px !important;
}
.stButton button { border-radius: 12px !important; }
[data-testid="stFileUploader"] { background: #FFFCF7; border-radius: 16px; padding: 8px; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


def label_of(col: str) -> str:
    return FEATURE_LABELS.get(col, col.replace("_", " ").title())


def safe_int(value, default, lo=None, hi=None):
    try:
        if value is None or pd.isna(value):
            number = int(default)
        else:
            number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        number = int(default)
    if lo is not None:
        number = max(lo, number)
    if hi is not None:
        number = min(hi, number)
    return number


def safe_float(value, default, lo=None, hi=None):
    try:
        if value is None or pd.isna(value):
            number = float(default)
        else:
            number = float(value)
        if not np.isfinite(number):
            number = float(default)
    except (TypeError, ValueError, OverflowError):
        number = float(default)
    if lo is not None:
        number = max(float(lo), number)
    if hi is not None:
        number = min(float(hi), number)
    return float(number)


def style_fig(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(252,250,245,0.65)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color=MUTED, size=13),
        title=dict(font=dict(family="Fraunces, serif", size=18, color=INK), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=8, t=56, b=8),
        colorway=[TEAL, AMBER, NAVY, "#A16207", RISK],
        xaxis=dict(gridcolor="#EDE7DC", zerolinecolor="#E7E1D6", linecolor="#E7E1D6"),
        yaxis=dict(gridcolor="#EDE7DC", zerolinecolor="#E7E1D6", linecolor="#E7E1D6"),
        hoverlabel=dict(bgcolor=CARD, font_color=INK, font_family="Plus Jakarta Sans, sans-serif"),
    )
    return fig


def unique_sorted(series):
    values = [str(v) for v in series.dropna().astype(str).tolist() if str(v).strip()]
    return sorted(set(values), key=str.lower)


def option_index(options, value, fallback=0):
    if value in options:
        return options.index(value)
    return fallback


def align_csv_columns(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    lower_map = {}
    for col in cleaned.columns:
        key = col.lower()
        if key not in lower_map:
            lower_map[key] = col
    rename = {}
    for feature in features:
        if feature in cleaned.columns:
            continue
        match = lower_map.get(feature.lower())
        if match:
            rename[match] = feature
    if rename:
        cleaned = cleaned.rename(columns=rename)
    return cleaned


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return (cleaned[:60] or "startup")


def apply_plotly_template():
    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = [TEAL, AMBER, NAVY, "#A16207"]


@st.cache_resource
def load_assets():
    missing = [path for path in [DATA, METRICS, REPORT, *MODELS.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files: " + ", ".join(path.name for path in missing)
        )
    models = {name: joblib.load(path) for name, path in MODELS.items()}
    data = pd.read_excel(DATA, sheet_name="Expanded_OnePlus_Round")
    data = data[data["model_label"].notna()].copy()
    if data.empty:
        raise ValueError("The research dataset has no labelled startups to display.")
    metrics = pd.read_csv(METRICS)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    primary = models["Logistic Regression"]
    features = list(primary.named_steps["preprocess"].feature_names_in_)
    present = [col for col in features if col in data.columns]
    missing_cols = [col for col in features if col not in data.columns]
    if missing_cols:
        raise ValueError("Dataset is missing model features: " + ", ".join(missing_cols))
    X = data[present]
    for name, model in models.items():
        try:
            data[f"score__{name}"] = model.predict_proba(X)[:, 1]
        except Exception:
            data[f"score__{name}"] = np.nan
    return models, data, metrics, report, features


apply_plotly_template()

try:
    models, data, metrics, report, features = load_assets()
except Exception as exc:
    st.error(f"StartupLens could not start: {exc}")
    st.caption("Confirm the Excel dataset, metrics CSV, evaluation report, and joblib model files are in this folder.")
    st.stop()

primary = models["Logistic Regression"]
numeric_cols = [c for c in features if c in data.columns and pd.api.types.is_numeric_dtype(data[c])]
cat_cols = [c for c in features if c not in numeric_cols]
labelled = int(len(data))
failed_n = int((data["model_label"] == 1).sum()) if "model_label" in data.columns else 0
operating_n = labelled - failed_n

st.markdown(
    f"""
<div class="sl-hero">
  <div>
    <div style="display:flex;gap:12px;align-items:center;margin-bottom:10px;">
      <div class="sl-mark">SL</div>
      <p class="sl-kicker">Investor intelligence</p>
    </div>
    <h1>Screen startups on observed funding trajectories</h1>
    <p>StartupLens compares a company’s funding path with a research cohort of {labelled} labelled startups ({failed_n} failed, {operating_n} operating). It supports due diligence. It does not replace it.</p>
  </div>
  <div class="sl-chip-row">
    <span class="sl-chip">Primary model · Logistic Regression</span>
    <span class="sl-chip">Fixed comparison threshold · 0.50</span>
    <span class="sl-chip">4-fold stratified CV</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

sample_names = unique_sorted(data["company_name"]) if "company_name" in data.columns else []
preferred_sample = "Redox" if "Redox" in sample_names else (sample_names[0] if sample_names else "")

with st.sidebar:
    st.markdown("### Screening controls")
    selected_model = st.selectbox("Prediction model", list(models.keys()), index=0)
    threshold = st.slider(
        "Classification threshold",
        0.05,
        0.95,
        TARGET_THRESHOLD,
        0.05,
        help="A lower threshold flags more startups as higher risk and can increase false alarms. Published comparison metrics stay fixed at 0.50.",
    )
    st.caption("Scores are uncalibrated model outputs, not real-world probabilities.")
    st.divider()
    st.markdown("### Example profile")
    sample_name = st.selectbox(
        "Load a company from the research dataset",
        ["Custom startup"] + sample_names,
        index=(sample_names.index(preferred_sample) + 1 if preferred_sample else 0),
        help="Loads historical features from the project dataset. This is not a live company update.",
    )
    st.markdown("**Workflow**")
    st.markdown("1. Load an example or enter a profile\n2. Optionally score a CSV batch\n3. Generate the assessment\n4. Compare models and inspect limitations")

sample_row = None
if sample_name != "Custom startup" and "company_name" in data.columns:
    matches = data[data["company_name"].astype(str) == sample_name]
    if not matches.empty:
        sample_row = matches.iloc[0]

cohort_num = data[numeric_cols].median(numeric_only=True).to_dict()
cohort_cat = {}
for col in cat_cols:
    modes = data[col].dropna().mode() if col in data.columns else pd.Series(dtype=object)
    cohort_cat[col] = str(modes.iloc[0]) if len(modes) else ""
num_defaults = sample_row[numeric_cols].to_dict() if sample_row is not None else cohort_num
cat_defaults = {}
for col in cat_cols:
    if sample_row is not None and pd.notna(sample_row.get(col)):
        cat_defaults[col] = str(sample_row.get(col))
    else:
        cat_defaults[col] = cohort_cat.get(col, "")

country_opts = [""] + unique_sorted(data["country"]) if "country" in data.columns else [""]
sector_opts = [""] + unique_sorted(data["sector_name"]) if "sector_name" in data.columns else [""]
city_opts = [""] + unique_sorted(data["city"]) if "city" in data.columns else [""]
stage_opts = [""] + unique_sorted(data["latest_observed_stage"]) if "latest_observed_stage" in data.columns else [""]

tab_assess, tab_compare, tab_metrics, tab_guide = st.tabs(
    ["Startup assessment", "Compare models", "Model performance", "How it works"]
)


def score_frame(model, frame: pd.DataFrame) -> np.ndarray:
    aligned = frame.reindex(columns=features)
    for col in numeric_cols:
        if col in aligned.columns:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
    probs = model.predict_proba(aligned)[:, 1]
    return np.clip(np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)


def sample_year(sample, column, fallback):
    if sample is None or column not in sample.index or pd.isna(sample.get(column)):
        return fallback
    try:
        return int(pd.to_datetime(sample.get(column)).year)
    except Exception:
        return fallback


def build_profile(
    *,
    country,
    city,
    sector,
    stage,
    founded,
    rounds,
    dated_rounds,
    disclosed_rounds,
    total,
    avg_amount,
    median_amount,
    largest,
    funding_start,
    funding_end,
    avg_gap,
    median_gap,
    last_gap,
    prior_gap,
    gap_std,
    growth,
    avg_investors,
    max_investors,
    stages,
    advances,
    mentions,
    amount_share,
    slowdown,
    investor_trend,
    base_row=None,
):
    warnings = []
    if funding_end < funding_start:
        warnings.append("Last observed funding year was before the first year, so the years were swapped.")
        funding_start, funding_end = funding_end, funding_start
    if disclosed_rounds > rounds:
        warnings.append("Disclosed-amount rounds exceed observed rounds. The disclosed count was capped.")
        disclosed_rounds = rounds
    if dated_rounds > rounds:
        warnings.append("Dated rounds exceed observed rounds. The dated count was capped.")
        dated_rounds = rounds
    if total > 0 and largest > total:
        warnings.append("Largest round exceeds total observed funding. Largest round was capped at total funding.")
        largest = total
    if amount_share > 1:
        amount_share = 1.0
        warnings.append("Share of rounds with disclosed amounts was clipped to 1.0.")
    span_days = max(0, (int(funding_end) - int(funding_start)) * 365)
    age_first = max(0, int(funding_start) - int(founded))
    age_last = max(0, int(funding_end) - int(founded))
    if age_last < age_first:
        age_last = age_first
        warnings.append("Age at last round was below age at first round; last age was raised to match.")
    year_span = max(1, int(funding_end) - int(funding_start) + 1)
    rounds_per_year = rounds / year_span

    if base_row is not None:
        vals = {k: base_row[k] if k in base_row.index else np.nan for k in features}
    else:
        vals = {k: np.nan for k in features}
        vals["investor_count_change_first_to_last"] = 0.0
        vals["investor_count_growth_first_to_last"] = 0.0

    vals.update(
        {
            "country": country or np.nan,
            "city": city or np.nan,
            "sector_name": sector or np.nan,
            "latest_observed_stage": stage or np.nan,
            "founded_year": founded,
            "observed_funding_rounds": rounds,
            "dated_funding_rounds": dated_rounds,
            "rounds_with_disclosed_amount": disclosed_rounds,
            "observed_total_funding_usd": total,
            "average_round_amount_usd": avg_amount,
            "median_round_amount_usd": median_amount,
            "largest_round_amount_usd": largest,
            "average_days_between_observed_rounds": avg_gap,
            "median_days_between_observed_rounds": median_gap,
            "funding_gap_std_days": gap_std,
            "disclosed_amount_growth_first_to_last": growth,
            "distinct_funding_stages": stages,
            "average_investor_count_per_round": avg_investors,
            "max_investor_count_in_round": max_investors,
            "total_news_mentions": mentions,
            "last_funding_gap_days": last_gap,
            "prior_funding_gap_median_days": prior_gap,
            "funding_slowdown_ratio": slowdown,
            "recent_vs_previous_round_amount_growth": growth,
            "amounts_disclosed_share": amount_share,
            "investor_participation_trend": investor_trend,
            "stage_advances_observed": advances,
        }
    )

    keep_sample_derived = False
    if base_row is not None:
        sample_founded = safe_int(base_row.get("founded_year"), founded)
        sample_start = sample_year(base_row, "first_observed_round_date", funding_start)
        sample_end = sample_year(base_row, "last_observed_round_date", funding_end)
        keep_sample_derived = (
            int(founded) == sample_founded
            and int(funding_start) == sample_start
            and int(funding_end) == sample_end
            and int(rounds) == safe_int(base_row.get("observed_funding_rounds"), rounds)
            and abs(float(total) - safe_float(base_row.get("observed_total_funding_usd"), total)) < 1
            and abs(float(largest) - safe_float(base_row.get("largest_round_amount_usd"), largest)) < 1
            and abs(float(last_gap) - safe_float(base_row.get("last_funding_gap_days"), last_gap)) < 0.5
            and abs(float(prior_gap) - safe_float(base_row.get("prior_funding_gap_median_days"), prior_gap)) < 0.5
        )

    if not keep_sample_derived:
        vals["observed_funding_span_days"] = span_days
        vals["company_age_at_first_observed_round_years"] = age_first
        vals["company_age_at_last_observed_round_years"] = age_last
        vals["funding_gap_change_days"] = last_gap - prior_gap
        vals["recent_round_amount_share_of_total"] = largest / total if total > 0 else np.nan
        vals["rounds_per_observed_funding_year"] = rounds_per_year
        if base_row is None:
            vals["highest_observed_stage_rank"] = stages
            vals["distinct_known_stage_levels"] = stages
        if base_row is not None:
            warnings.append("Derived timing and mix fields were recalculated from the years and amounts you entered.")
    else:
        vals["funding_gap_change_days"] = base_row.get("funding_gap_change_days", last_gap - prior_gap)

    row = pd.DataFrame([{k: vals.get(k, np.nan) for k in features}], columns=features)
    context = {
        "funding_start": int(funding_start),
        "funding_end": int(funding_end),
        "founded": int(founded),
        "rounds": int(rounds),
        "last_gap": float(last_gap),
        "total": float(total),
        "largest": float(largest),
        "mentions": int(mentions),
        "avg_gap": float(avg_gap),
        "used_research_row": base_row is not None,
        "kept_sample_derived": keep_sample_derived,
    }
    return row, context, warnings


def gauge_chart(score: float, threshold_value: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"family": "Fraunces, serif", "size": 36, "color": INK}},
            title={"text": "Uncalibrated failure score", "font": {"size": 14, "color": MUTED}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": MUTED},
                "bar": {"color": RISK if score >= threshold_value else TEAL, "thickness": 0.28},
                "bgcolor": "#F7F2EA",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold_value * 100], "color": "#D9EDE7"},
                    {"range": [threshold_value * 100, 100], "color": "#F8D7D4"},
                ],
                "threshold": {
                    "line": {"color": INK, "width": 3},
                    "thickness": 0.85,
                    "value": threshold_value * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif"),
    )
    return fig


def cohort_scatter(current_total, current_gap, current_name):
    plot_df = data.copy()
    plot_df["Outcome"] = np.where(plot_df["model_label"] == 1, "Failed", "Operating")
    plot_df["Total funding (USD)"] = pd.to_numeric(plot_df["observed_total_funding_usd"], errors="coerce")
    plot_df["Days since last round"] = pd.to_numeric(plot_df["last_funding_gap_days"], errors="coerce")
    fig = px.scatter(
        plot_df.dropna(subset=["Total funding (USD)", "Days since last round"]),
        x="Total funding (USD)",
        y="Days since last round",
        color="Outcome",
        hover_name="company_name" if "company_name" in plot_df.columns else None,
        color_discrete_map={"Operating": TEAL, "Failed": RISK},
        opacity=0.7,
        title="Where this profile sits in the research cohort",
        log_x=True,
    )
    fig.add_trace(
        go.Scatter(
            x=[max(current_total, 1)],
            y=[current_gap],
            mode="markers+text",
            name="Assessed startup",
            text=[current_name],
            textposition="top center",
            marker=dict(size=14, color=AMBER, symbol="diamond", line=dict(width=1, color=INK)),
        )
    )
    return style_fig(fig, 400)


with tab_assess:
    st.subheader("Validation snapshot")
    st.caption("These figures come from the research dataset, shown before you score a new startup. Four-fold stratified cross-validation; out-of-fold predictions; threshold fixed at 0.50.")
    lr_metrics = metrics[metrics["model"].astype(str).str.contains("Logistic", case=False, na=False)]
    if lr_metrics.empty:
        st.warning("Logistic Regression metrics were not found in model_comparison_metrics.csv.")
    else:
        lm = lr_metrics.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{safe_float(lm.get('accuracy'), 0):.1%}")
        m2.metric("Balanced accuracy", f"{safe_float(lm.get('balanced_accuracy'), 0):.1%}")
        m3.metric("Failure recall", f"{safe_float(lm.get('recall_failure'), 0):.1%}")
        m4.metric("PR-AUC", f"{safe_float(lm.get('pr_auc_average_precision'), 0):.3f}")
        st.caption(
            "Logistic Regression is the primary screening model because it holds 95%+ accuracy while also producing non-zero failure recall, failure precision and PR-AUC. Accuracy alone is misleading with only "
            f"{failed_n} failed startups in the {labelled}-company dataset."
        )

    with st.expander("View validation metrics for all models", expanded=False):
        if metrics.empty:
            st.info("No comparison metrics are available.")
        else:
            display = metrics.rename(columns={**METRIC_LABELS, "model": "Model"})
            cols = ["Model"] + [METRIC_LABELS[k] for k in METRIC_LABELS if METRIC_LABELS[k] in display.columns]
            format_map = {METRIC_LABELS[k]: "{:.1%}" for k in METRIC_LABELS if k != "pr_auc_average_precision"}
            format_map["PR-AUC"] = "{:.3f}"
            st.dataframe(
                display[cols].style.format({c: format_map[c] for c in cols if c in format_map}),
                hide_index=True,
                use_container_width=True,
            )

    st.divider()
    st.subheader("Batch-score a CSV")
    st.write(
        "Upload one or more startup records. Extra columns such as company name are kept. Unknown numeric values can be left blank; the saved pipeline imputes them."
    )
    template = pd.DataFrame(columns=["company_name"] + features)
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False),
        file_name="startuplens_test_dataset_template.csv",
        mime="text/csv",
    )
    uploaded_csv = st.file_uploader("Choose a test CSV file", type=["csv"], key="startup_test_csv")
    if uploaded_csv is not None:
        try:
            test_df = pd.read_csv(uploaded_csv)
            if test_df.empty:
                st.warning("The uploaded CSV has no data rows.")
            else:
                test_df = align_csv_columns(test_df, features + ["company_name"])
                missing_features = [col for col in features if col not in test_df.columns]
                if missing_features:
                    st.error("This CSV is missing required model input columns: " + ", ".join(missing_features))
                    st.caption("Download the template, keep the header names, and leave unknown values blank.")
                else:
                    b1, b2 = st.columns(2)
                    with b1:
                        batch_model_name = st.selectbox("Model for CSV predictions", list(models.keys()), index=0, key="batch_model_choice")
                    with b2:
                        batch_threshold = st.slider("CSV failure-flag threshold", 0.05, 0.95, TARGET_THRESHOLD, 0.05, key="batch_threshold")
                    batch_model = models[batch_model_name]
                    batch_scores = score_frame(batch_model, test_df)
                    batch_results = test_df.copy()
                    if "company_name" not in batch_results.columns:
                        batch_results.insert(0, "company_name", [f"Row {i+1}" for i in range(len(batch_results))])
                    batch_results["company_name"] = batch_results["company_name"].fillna("").astype(str).replace({"": np.nan})
                    batch_results["company_name"] = batch_results["company_name"].fillna(
                        pd.Series([f"Row {i+1}" for i in range(len(batch_results))], index=batch_results.index)
                    )
                    batch_results["failure_score_uncalibrated"] = batch_scores
                    batch_results["threshold"] = batch_threshold
                    batch_results["prediction"] = np.where(batch_scores >= batch_threshold, "Failed", "Operating")
                    flagged = int((batch_results["prediction"] == "Failed").sum())
                    st.success(f"Scored {len(batch_results)} startup record(s) using {batch_model_name}. {flagged} flagged as Failed at {batch_threshold:.0%}.")
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Records scored", f"{len(batch_results)}")
                    k2.metric("Flagged Failed", f"{flagged}")
                    k3.metric("Median score", f"{float(np.median(batch_scores)):.1%}")
                    preview_cols = ["company_name", "failure_score_uncalibrated", "threshold", "prediction"]
                    st.dataframe(
                        batch_results[preview_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "failure_score_uncalibrated": st.column_config.ProgressColumn("Failure score", min_value=0, max_value=1, format="%.1f%%"),
                        },
                    )
                    hist = px.histogram(
                        batch_results,
                        x="failure_score_uncalibrated",
                        nbins=min(20, max(5, len(batch_results))),
                        title="Score distribution for the uploaded file",
                        color="prediction",
                        color_discrete_map={"Operating": TEAL, "Failed": RISK},
                    )
                    hist.update_xaxes(range=[0, 1], title="Uncalibrated failure score")
                    hist.update_yaxes(title="Startups")
                    st.plotly_chart(style_fig(hist, 320), use_container_width=True)
                    st.download_button(
                        "Download CSV with predictions",
                        data=batch_results.to_csv(index=False).encode("utf-8"),
                        file_name="startuplens_test_predictions.csv",
                        mime="text/csv",
                    )
                    st.caption("Validation metrics above are not recalculated on your CSV unless it contains verified outcome labels.")
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")
            st.caption("Use a comma-separated file whose column names match the template.")

    st.divider()
    st.subheader("Build a startup profile")
    st.write("Enter what you know. Blank categoricals and typical numeric defaults are passed to the saved preprocessing pipeline.")
    with st.form("startup_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            company = st.text_input(
                "Startup name",
                sample_name if sample_name != "Custom startup" else "New startup",
            )
            country = st.selectbox("Country", country_opts, index=option_index(country_opts, cat_defaults.get("country", "")))
            sector = st.selectbox("Sector", sector_opts, index=option_index(sector_opts, cat_defaults.get("sector_name", "")))
        with c2:
            founded = st.number_input("Founded year", min_value=1980, max_value=2026, value=safe_int(num_defaults.get("founded_year"), 2018, 1980, 2026))
            city_val = str(sample_row.get("city", "")) if sample_row is not None and pd.notna(sample_row.get("city")) else ""
            city = st.selectbox("City", city_opts, index=option_index(city_opts, city_val))
            stage = st.selectbox("Latest funding stage", stage_opts, index=option_index(stage_opts, cat_defaults.get("latest_observed_stage", "")))
        with c3:
            rounds = st.number_input("Observed funding rounds", 0, 30, safe_int(num_defaults.get("observed_funding_rounds"), 2, 0, 30))
            if sample_row is not None and pd.notna(sample_row.get("first_observed_round_date")):
                try:
                    sample_start = int(pd.to_datetime(sample_row.get("first_observed_round_date")).year)
                except Exception:
                    sample_start = founded
            else:
                sample_start = founded
            funding_start = st.number_input("First observed funding year", 1980, 2026, safe_int(sample_start, founded, 1980, 2026))
            if sample_row is not None and pd.notna(sample_row.get("last_observed_round_date")):
                try:
                    sample_end = int(pd.to_datetime(sample_row.get("last_observed_round_date")).year)
                except Exception:
                    sample_end = 2025
            else:
                sample_end = 2025
            funding_end = st.number_input("Last observed funding year", 1980, 2026, safe_int(sample_end, 2025, 1980, 2026))

        st.markdown("#### Funding trajectory")
        st.caption("Amounts are in USD. Time gaps are in days. Ratios are decimals (0.25 = 25%).")
        a, b, c, d = st.columns(4)
        with a:
            total = st.number_input("Total observed funding (USD)", 0.0, value=safe_float(num_defaults.get("observed_total_funding_usd"), 0, 0))
            avg_amount = st.number_input("Average round amount (USD)", 0.0, value=safe_float(num_defaults.get("average_round_amount_usd"), 0, 0))
            median_amount = st.number_input("Median round amount (USD)", 0.0, value=safe_float(num_defaults.get("median_round_amount_usd"), 0, 0))
        with b:
            largest = st.number_input("Largest round amount (USD)", 0.0, value=safe_float(num_defaults.get("largest_round_amount_usd"), 0, 0))
            disclosed_rounds = st.number_input("Rounds with disclosed amount", 0, 30, safe_int(num_defaults.get("rounds_with_disclosed_amount"), 1, 0, 30))
            dated_rounds = st.number_input("Rounds with known dates", 0, 30, safe_int(num_defaults.get("dated_funding_rounds"), 2, 0, 30))
        with c:
            avg_gap = st.number_input("Average days between rounds", 0.0, value=safe_float(num_defaults.get("average_days_between_observed_rounds"), 0, 0))
            median_gap = st.number_input("Median days between rounds", 0.0, value=safe_float(num_defaults.get("median_days_between_observed_rounds"), 0, 0))
            last_gap = st.number_input("Days since last funding round", 0.0, value=safe_float(num_defaults.get("last_funding_gap_days"), 0, 0))
        with d:
            prior_gap = st.number_input("Prior median funding gap (days)", 0.0, value=safe_float(num_defaults.get("prior_funding_gap_median_days"), 0, 0))
            gap_std = st.number_input("Variation in funding gaps (days)", 0.0, value=safe_float(num_defaults.get("funding_gap_std_days"), 0, 0))
            growth = st.number_input("First-to-last round amount growth", value=safe_float(num_defaults.get("disclosed_amount_growth_first_to_last"), 0))

        st.markdown("#### Investor and progression signals")
        e, f, g, h = st.columns(4)
        with e:
            avg_investors = st.number_input("Average investors per round", 0.0, value=safe_float(num_defaults.get("average_investor_count_per_round"), 0, 0))
            max_investors = st.number_input("Maximum investors in a round", 0, 100, safe_int(num_defaults.get("max_investor_count_in_round"), 0, 0, 100))
        with f:
            stages = st.number_input("Distinct funding stages", 0, 15, safe_int(num_defaults.get("distinct_funding_stages"), 0, 0, 15))
            advances = st.number_input("Observed stage advances", 0, 15, safe_int(num_defaults.get("stage_advances_observed"), 0, 0, 15))
        with g:
            mentions = st.number_input("Total news mentions", 0, 100000, safe_int(num_defaults.get("total_news_mentions"), 0, 0, 100000))
            amount_share = st.number_input(
                "Share of rounds with disclosed amounts",
                0.0,
                1.0,
                safe_float(num_defaults.get("amounts_disclosed_share"), 0, 0, 1),
                0.01,
            )
        with h:
            slowdown = st.number_input("Funding slowdown ratio", 0.0, value=safe_float(num_defaults.get("funding_slowdown_ratio"), 0, 0))
            investor_trend = st.number_input("Investor participation trend", value=safe_float(num_defaults.get("investor_participation_trend"), 0))
        submitted = st.form_submit_button("Analyse startup", type="primary", use_container_width=True)

    if submitted:
        row, context, warnings = build_profile(
            country=country,
            city=city,
            sector=sector,
            stage=stage,
            founded=founded,
            rounds=rounds,
            dated_rounds=dated_rounds,
            disclosed_rounds=disclosed_rounds,
            total=total,
            avg_amount=avg_amount,
            median_amount=median_amount,
            largest=largest,
            funding_start=funding_start,
            funding_end=funding_end,
            avg_gap=avg_gap,
            median_gap=median_gap,
            last_gap=last_gap,
            prior_gap=prior_gap,
            gap_std=gap_std,
            growth=growth,
            avg_investors=avg_investors,
            max_investors=max_investors,
            stages=stages,
            advances=advances,
            mentions=mentions,
            amount_share=amount_share,
            slowdown=slowdown,
            investor_trend=investor_trend,
            base_row=sample_row,
        )
        st.session_state["startup_row"] = row
        st.session_state["startup_name"] = (company or "").strip() or "Unnamed startup"
        st.session_state["profile_context"] = context
        st.session_state["profile_warnings"] = warnings

    if st.session_state.get("startup_row") is not None:
        row = st.session_state["startup_row"]
        name = st.session_state.get("startup_name", "Startup")
        context = st.session_state.get("profile_context") or {}
        warnings = st.session_state.get("profile_warnings") or []
        if not context:
            founded_fallback = safe_int(row.iloc[0].get("founded_year"), 2018)
            age_first = safe_int(row.iloc[0].get("company_age_at_first_observed_round_years"), 0)
            age_last = safe_int(row.iloc[0].get("company_age_at_last_observed_round_years"), 0)
            context = {
                "funding_start": founded_fallback + age_first,
                "funding_end": founded_fallback + max(age_first, age_last),
                "founded": founded_fallback,
                "rounds": safe_int(row.iloc[0].get("observed_funding_rounds"), 0),
                "last_gap": safe_float(row.iloc[0].get("last_funding_gap_days"), 0, 0),
                "total": safe_float(row.iloc[0].get("observed_total_funding_usd"), 0, 0),
                "largest": safe_float(row.iloc[0].get("largest_round_amount_usd"), 0, 0),
                "mentions": safe_int(row.iloc[0].get("total_news_mentions"), 0),
                "avg_gap": safe_float(row.iloc[0].get("average_days_between_observed_rounds"), 0, 0),
            }

        st.divider()
        st.subheader(f"Assessment · {name}")
        if context.get("used_research_row") and context.get("kept_sample_derived"):
            st.caption("This score starts from the saved research-row features for the loaded company, then applies any form edits.")
        elif context.get("used_research_row"):
            st.caption("This score starts from the loaded research company, with timing and mix fields recalculated from the years and amounts you entered.")
        for note in warnings:
            st.warning(note)

        try:
            chosen = models[selected_model]
            score = float(score_frame(chosen, row)[0])
        except Exception as exc:
            st.error(f"The selected model could not score this profile: {exc}")
            st.stop()

        flag = score >= threshold
        cohort_col = f"score__{selected_model}"
        if cohort_col in data.columns and data[cohort_col].notna().any():
            percentile = float((data[cohort_col].dropna() < score).mean())
        else:
            percentile = None

        result_left, result_right = st.columns([1.05, 1.35])
        with result_left:
            st.plotly_chart(gauge_chart(score, threshold), use_container_width=True)
        with result_right:
            r1, r2, r3 = st.columns(3)
            r1.metric("Model", selected_model)
            r2.metric("Failure score", f"{score:.1%}")
            r3.metric("Decision", "Failed" if flag else "Operating")
            if percentile is not None:
                st.caption(f"This score is higher than {percentile:.0%} of the {labelled}-company research cohort under {selected_model}.")
            if flag:
                st.markdown(
                    "<div class='sl-flag-risk'><h4>Flagged as Failed at the current threshold</h4><p>This is a model prediction, not a verified operating status. Continue due diligence before any investment decision.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='sl-flag-ok'><h4>Flagged as Operating at the current threshold</h4><p>This is a model prediction, not a guarantee that the startup is operating or will remain successful.</p></div>",
                    unsafe_allow_html=True,
                )
            st.progress(min(1.0, max(0.0, score)), text=f"Score {score:.1%} · threshold {threshold:.0%}")

        st.markdown("#### Why the score moved")
        st.caption("One-feature-at-a-time sensitivity against typical training-cohort values. This describes model behaviour, not causation.")
        base_score = score
        sensitivity = []
        for col in features:
            alt = row.copy()
            if col in numeric_cols:
                alt.loc[0, col] = cohort_num.get(col, 0)
            else:
                alt.loc[0, col] = cohort_cat.get(col, np.nan)
            try:
                alt_score = float(score_frame(chosen, alt)[0])
                delta = base_score - alt_score
                if abs(delta) > 0.0005:
                    sensitivity.append({"Feature": label_of(col), "column": col, "delta": delta})
            except Exception:
                continue
        sensitivity.sort(key=lambda item: abs(item["delta"]), reverse=True)
        if sensitivity:
            explain = pd.DataFrame(sensitivity[:8])
            explain["Direction"] = np.where(explain["delta"] > 0, "Raises score", "Lowers score")
            bar = px.bar(
                explain.sort_values("delta"),
                x="delta",
                y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={"Raises score": RISK, "Lowers score": TEAL},
                title="Largest score changes vs cohort-typical values",
            )
            bar.update_xaxes(title="Change in failure score", tickformat=".1%")
            bar.update_yaxes(title="")
            st.plotly_chart(style_fig(bar, 360), use_container_width=True)
            st.dataframe(
                explain.rename(columns={"delta": "Change vs typical value"})[["Feature", "Change vs typical value", "Direction"]],
                hide_index=True,
                use_container_width=True,
                column_config={"Change vs typical value": st.column_config.NumberColumn(format="%.1%")},
            )
        else:
            st.info("No large one-feature changes were detected relative to the training-cohort reference values.")

        st.markdown("#### All three models on this profile")
        comp = []
        for model_name, model in models.items():
            try:
                p = float(score_frame(model, row)[0])
            except Exception:
                p = np.nan
            comp.append(
                {
                    "Model": model_name,
                    "Failure score (uncalibrated)": p,
                    "Classification": "Failed" if pd.notna(p) and p >= threshold else "Operating",
                }
            )
        comp_df = pd.DataFrame(comp).sort_values("Failure score (uncalibrated)", ascending=False)
        bars = px.bar(
            comp_df,
            x="Model",
            y="Failure score (uncalibrated)",
            color="Model",
            range_y=[0, 1],
            title="Predicted failure scores",
            text=comp_df["Failure score (uncalibrated)"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—"),
        )
        bars.add_hline(y=threshold, line_dash="dot", line_color=INK, annotation_text=f"Threshold {threshold:.0%}")
        bars.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(style_fig(bars, 360), use_container_width=True)
        st.dataframe(
            comp_df,
            hide_index=True,
            use_container_width=True,
            column_config={"Failure score (uncalibrated)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.1f%%")},
        )

        st.download_button(
            "Download assessment as CSV",
            row.assign(company_name=name, model_score=score, model=selected_model, threshold=threshold).to_csv(index=False),
            file_name=f"{sanitize_filename(name)}_startuplens_assessment.csv",
            mime="text/csv",
        )

        vis1, vis2 = st.columns(2)
        with vis1:
            st.markdown("#### Observation window")
            start_year = int(context["funding_start"])
            end_year = int(context["funding_end"])
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[start_year, end_year],
                    y=["Observed window", "Observed window"],
                    mode="lines+markers",
                    line=dict(width=14, color=TEAL),
                    marker=dict(size=16, color=NAVY),
                    hovertemplate="Year %{x}<extra></extra>",
                    name="Entered window",
                )
            )
            fig.update_yaxes(title="")
            fig.update_xaxes(title="Year", dtick=1)
            fig.update_layout(title="First to last observed funding year")
            st.caption("The source file stores company-level aggregates, not a verified round-by-round event log. This window only reflects the years you entered.")
            st.plotly_chart(style_fig(fig, 280), use_container_width=True)
        with vis2:
            st.markdown("#### Cohort context")
            st.plotly_chart(
                cohort_scatter(context.get("total", 0), context.get("last_gap", 0), name),
                use_container_width=True,
            )

        st.markdown("#### What-if scenario")
        st.caption("Change selected inputs and see how the chosen model score responds, holding other inputs constant.")
        w1, w2 = st.columns(2)
        with w1:
            scenario_gap = st.slider("Scenario: days since last funding", 0, 1800, int(np.clip(context.get("last_gap", 0), 0, 1800)), 1)
        with w2:
            scenario_rounds = st.slider("Scenario: observed funding rounds", 0, 30, int(np.clip(context.get("rounds", 0), 0, 30)), 1)
        scenario_row = row.copy()
        if "last_funding_gap_days" in scenario_row.columns:
            scenario_row.loc[0, "last_funding_gap_days"] = scenario_gap
        if "observed_funding_rounds" in scenario_row.columns:
            scenario_row.loc[0, "observed_funding_rounds"] = scenario_rounds
        if "prior_funding_gap_median_days" in scenario_row.columns:
            scenario_row.loc[0, "funding_gap_change_days"] = scenario_gap - safe_float(scenario_row.loc[0].get("prior_funding_gap_median_days"), 0)
        try:
            scenario_score = float(score_frame(chosen, scenario_row)[0])
            st.metric("What-if failure score", f"{scenario_score:.1%}", delta=f"{scenario_score - score:+.1%} vs original", delta_color="inverse")
        except Exception as exc:
            st.warning(f"This scenario could not be scored: {exc}")
        st.caption("Scenario scores are model outputs, not forecasts. Extreme combinations may fall outside the training data.")

        st.markdown("#### Investor due-diligence checklist")
        st.caption("Items for human verification. These are not currently evaluated by the model.")
        diligence_items = [
            "Verify funding announcements, dates and amounts from primary sources",
            "Review cash runway, burn rate and current liquidity",
            "Assess revenue growth, customer retention and unit economics",
            "Review founder experience, governance and cap table",
            "Evaluate market size, competition and regulatory exposure",
            "Confirm investor commitments and terms of the latest round",
        ]
        dcols = st.columns(2)
        for idx, item in enumerate(diligence_items):
            dcols[idx % 2].checkbox(item, key=f"diligence_{idx}")

        st.markdown("#### Downloadable investor report")
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

            pdf_buf = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buf, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
            styles = getSampleStyleSheet()
            story = [
                Paragraph("StartupLens investor screening report", styles["Title"]),
                Paragraph(f"<b>Startup:</b> {name}", styles["Normal"]),
                Paragraph(f"<b>Selected model:</b> {selected_model}", styles["Normal"]),
                Paragraph(f"<b>Failure score (uncalibrated):</b> {score:.1%}", styles["Normal"]),
                Paragraph(f"<b>Decision threshold:</b> {threshold:.0%}", styles["Normal"]),
                Paragraph(f"<b>Decision:</b> {'Failed' if flag else 'Operating'}", styles["Normal"]),
                Spacer(1, 8),
                Paragraph("Validation metrics — four-fold stratified cross-validation; fixed threshold 0.50", styles["Heading2"]),
            ]
            rows = [["Model", "Accuracy", "Bal. acc.", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]]
            for _, metric_row in metrics.iterrows():
                rows.append(
                    [
                        str(metric_row.get("model", "")),
                        f"{safe_float(metric_row.get('accuracy'), 0):.1%}",
                        f"{safe_float(metric_row.get('balanced_accuracy'), 0):.1%}",
                        f"{safe_float(metric_row.get('precision_failure'), 0):.1%}",
                        f"{safe_float(metric_row.get('recall_failure'), 0):.1%}",
                        f"{safe_float(metric_row.get('f1_failure'), 0):.3f}",
                        f"{safe_float(metric_row.get('roc_auc'), 0):.3f}",
                        f"{safe_float(metric_row.get('pr_auc_average_precision'), 0):.3f}",
                    ]
                )
            tbl = Table(rows, repeatRows=1)
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0E7C66")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D6D0C6")),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFCF7"), colors.HexColor("#F4F1EA")]),
                    ]
                )
            )
            story += [
                tbl,
                Spacer(1, 8),
                Paragraph("Limitations", styles["Heading2"]),
                Paragraph(
                    "The filtered dataset contains 169 startups, including only 8 failed cases. Scores are not calibrated real-world probabilities. Funding history does not establish a strict historical prediction cutoff. This report supports due diligence and is not investment advice.",
                    styles["BodyText"],
                ),
            ]
            doc.build(story)
            st.download_button(
                "Download PDF investor report",
                pdf_buf.getvalue(),
                file_name=f"{sanitize_filename(name)}_StartupLens_Report.pdf",
                mime="application/pdf",
            )
        except ImportError:
            st.warning("PDF generation requires reportlab. Install dependencies from requirements.txt and restart the app.")
        except Exception as exc:
            st.warning(f"The PDF report could not be generated: {exc}")


with tab_compare:
    st.subheader("Compare model behaviour")
    st.write("All three models score the exact same startup profile. Outputs can differ because they learn different patterns.")
    if st.session_state.get("startup_row") is None:
        st.info("Complete a startup assessment first. The model comparison will appear here.")
    else:
        row = st.session_state["startup_row"]
        rows = []
        for name, model in models.items():
            try:
                p = float(score_frame(model, row)[0])
            except Exception:
                p = np.nan
            rows.append({"Model": name, "Failure score": p, "At current threshold": "Failed" if pd.notna(p) and p >= threshold else "Operating"})
        dfp = pd.DataFrame(rows)
        fig = px.bar(dfp, x="Model", y="Failure score", color="Model", range_y=[0, 1], title="Predicted failure scores (not calibrated probabilities)")
        fig.add_hline(y=threshold, line_dash="dot", line_color=INK, annotation_text=f"Threshold {threshold:.0%}")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        st.dataframe(dfp, hide_index=True, use_container_width=True)
    st.markdown("**How the models differ**")
    st.markdown(
        "- **Logistic Regression:** estimates a linear relationship between transformed features and the log-odds of failure. It is the primary screening model.\n"
        "- **Random Forest:** combines many decision trees trained on bootstrapped samples. It can capture nonlinear patterns.\n"
        "- **Extra Trees:** combines highly randomized decision trees and can behave differently from Random Forest on the same profile."
    )

with tab_metrics:
    st.subheader("Validation performance")
    st.caption(f"Four-fold stratified cross-validation · out-of-fold predictions · fixed 0.50 threshold · {labelled} startups, {failed_n} failures")
    if metrics.empty:
        st.warning("No metrics file rows were found.")
    else:
        display = metrics.rename(columns={**METRIC_LABELS, "model": "Model"})
        pretty_cols = ["Model"] + [METRIC_LABELS[k] for k in METRIC_LABELS]
        format_map = {METRIC_LABELS[k]: "{:.1%}" for k in METRIC_LABELS if k != "pr_auc_average_precision"}
        format_map["PR-AUC"] = "{:.3f}"
        st.dataframe(
            display[pretty_cols].style.format({c: format_map[c] for c in pretty_cols if c in format_map}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Why Logistic Regression is the primary model")
        st.write(
            "It is the only model in this comparison that reaches 95%+ accuracy while also recording non-zero failure recall, failure precision and a stronger PR-AUC. Random Forest and Extra Trees can match the high accuracy band, but they detect none of the failed companies at the reported 0.50 threshold."
        )
        chart = metrics.melt(
            id_vars="model",
            value_vars=list(METRIC_LABELS.keys()),
            var_name="Metric",
            value_name="Score",
        )
        chart["Metric"] = chart["Metric"].map(METRIC_LABELS)
        grouped = px.bar(chart, x="Metric", y="Score", color="model", barmode="group", range_y=[0, 1], title="Cross-validation metric comparison")
        st.plotly_chart(style_fig(grouped, 420), use_container_width=True)

        radar = go.Figure()
        theta = list(METRIC_LABELS.values())
        for _, metric_row in metrics.iterrows():
            values = [safe_float(metric_row.get(key), 0) for key in METRIC_LABELS]
            radar.add_trace(
                go.Scatterpolar(
                    r=values + values[:1],
                    theta=theta + theta[:1],
                    fill="toself",
                    name=str(metric_row.get("model", "Model")),
                    opacity=0.55,
                )
            )
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#EDE7DC"), bgcolor="rgba(0,0,0,0)"),
            title="Metric shape by model",
        )
        st.plotly_chart(style_fig(radar, 420), use_container_width=True)

        st.markdown("#### Confusion matrices at the 0.50 threshold")
        heat_cols = st.columns(min(3, max(1, len(metrics))))
        for idx, (_, metric_row) in enumerate(metrics.iterrows()):
            matrix = np.array(
                [
                    [safe_int(metric_row.get("true_negatives"), 0), safe_int(metric_row.get("false_positives"), 0)],
                    [safe_int(metric_row.get("false_negatives"), 0), safe_int(metric_row.get("true_positives"), 0)],
                ]
            )
            heat = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=["Pred. operating", "Pred. failed"],
                    y=["Actual operating", "Actual failed"],
                    colorscale=[[0, "#F4F1EA"], [1, TEAL]],
                    text=matrix,
                    texttemplate="%{text}",
                    showscale=False,
                )
            )
            heat.update_layout(title=str(metric_row.get("model", "Model")))
            heat_cols[idx % 3].plotly_chart(style_fig(heat, 280), use_container_width=True)
        st.warning(
            f"Only {failed_n} failures are available. Each validation fold contains about two failures, so a single changed prediction can move recall by roughly 12.5 percentage points. These estimates are highly uncertain."
        )

with tab_guide:
    st.subheader("What StartupLens does")
    st.markdown(
        "StartupLens uses observed company and funding-trajectory variables to estimate whether a startup resembles the operating or failed companies in the training dataset. The target is `model_label`: 0 = operating, 1 = failed."
    )
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Logistic Regression")
        st.write(
            "Despite the word “regression”, Logistic Regression is a classification algorithm here. It combines the input features, converts the result into a score between 0 and 1, and compares that score with a threshold. The score is not a calibrated real-world probability."
        )
        st.markdown("#### Threshold")
        st.write(
            "The default threshold is 0.50. The sidebar lets investors change it for scenario exploration. Reported model-comparison metrics remain fixed at 0.50. Lower thresholds generally catch more potential failures but can create more false alarms."
        )
    with g2:
        st.markdown("#### Key metrics")
        st.markdown(
            "- **Accuracy:** share of all predictions that are correct.\n"
            "- **Precision:** among startups flagged as failures, how many were actually failures.\n"
            "- **Recall:** among actual failures, how many the model caught.\n"
            "- **F1:** balance between precision and recall.\n"
            "- **Balanced accuracy:** average recall across operating and failed classes.\n"
            "- **ROC-AUC:** ranking quality across thresholds.\n"
            "- **PR-AUC:** precision-recall performance when failures are rare."
        )
    st.markdown("#### Important limitations")
    for item in report.get("limitations", []):
        st.markdown(f"- {item}")
    st.markdown("#### Why Logistic Regression is primary in this dataset")
    st.write(
        "It reaches 95%+ accuracy while still flagging at least one failed company at 0.50, with non-zero precision and a stronger PR-AUC than the tree models. Random Forest and Extra Trees remain near 95% accuracy by almost always predicting operating, so they miss all eight failed cases at this threshold."
    )
    st.info("Use the result as one input to due diligence. Validate financial statements, cash runway, customer traction, market conditions, governance and founder execution independently.")

st.divider()
st.caption("StartupLens · Academic prototype · Scores are uncalibrated model outputs, not investment advice or guarantees of future outcomes.")
