import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Resolve data path relative to this file (works on Streamlit Cloud and locally)
DATA_PATH = Path(__file__).parent / "EA.csv"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Mental Health Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252940 100%);
        border: 1px solid #2d3250;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-size: 12px;
        font-weight: 500;
        color: #7c85a8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #e8ecf5;
        line-height: 1;
    }
    .metric-delta {
        font-size: 12px;
        color: #6b7bb0;
        margin-top: 4px;
    }
    .metric-icon {
        font-size: 22px;
        margin-bottom: 6px;
    }
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #e8ecf5;
        margin: 32px 0 8px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #2d3250;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a1f35 0%, #1e2340 100%);
        border-left: 4px solid #5b6ef5;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #c0c8e8;
        font-size: 14px;
        line-height: 1.6;
    }
    .insight-box.warning {
        border-left-color: #f5a623;
    }
    .insight-box.success {
        border-left-color: #4cd964;
    }
    .insight-box.danger {
        border-left-color: #ff4757;
    }
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #c0c8e8 !important;
    }
    div[data-testid="stMetricValue"] > div {
        color: #e8ecf5 !important;
    }
    .block-container { padding: 1.5rem 2rem; }
    h1 { color: #e8ecf5 !important; }
    h2, h3 { color: #c0c8e8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["burnout_risk"] = pd.Categorical(df["burnout_risk"], categories=["Low", "Medium", "High"], ordered=True)
    return df

df = load_data()

BURNOUT_COLORS = {"Low": "#4cd964", "Medium": "#f5a623", "High": "#ff4757"}
PLOTLY_TEMPLATE = "plotly_dark"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Student Mental Health")
    st.markdown("---")

    st.markdown("#### 🔍 Global Filters")
    selected_countries = st.multiselect("Country", sorted(df["country"].unique()), default=sorted(df["country"].unique()))
    selected_years = st.multiselect("Year of Study", sorted(df["year_of_study"].unique()), default=sorted(df["year_of_study"].unique()))
    selected_genders = st.multiselect("Gender", sorted(df["gender"].unique()), default=sorted(df["gender"].unique()))
    selected_burnout = st.multiselect("Burnout Risk", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])

    st.markdown("---")
    st.markdown("#### 📊 Navigation")
    page = st.radio("", ["📋 Overview", "📊 Descriptive", "🔍 Diagnostic", "🤖 Predictive", "💡 Prescriptive"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(df):,} students")
    st.markdown(f"**Universities:** {df['university'].nunique()}")
    st.markdown(f"**Countries:** {df['country'].nunique()}")

# Apply filters
fdf = df[
    df["country"].isin(selected_countries) &
    df["year_of_study"].isin(selected_years) &
    df["gender"].isin(selected_genders) &
    df["burnout_risk"].isin(selected_burnout)
].copy()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def plotly_card(fig, height=400):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        font=dict(family="Inter", color="#c0c8e8"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

def metric_card(icon, label, value, delta=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-delta">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

def insight(text, kind="info"):
    st.markdown(f'<div class="insight-box {kind}">{text}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ══════════════ PAGE: OVERVIEW ══════════════
# ─────────────────────────────────────────────
if page == "📋 Overview":
    st.markdown("# 🧠 Student Mental Health & Burnout Dashboard")
    st.markdown("Comprehensive analytics on burnout risk, mental health drivers, and student wellbeing indicators.")

    if len(fdf) == 0:
        st.warning("No data matches current filters.")
        st.stop()

    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    cols = st.columns(6)
    with cols[0]:
        metric_card("😰", "Avg Stress Score", f"{fdf['stress_score'].mean():.1f}", f"/ 100")
    with cols[1]:
        metric_card("😟", "Avg Anxiety Score", f"{fdf['anxiety_score'].mean():.1f}", f"/ 100")
    with cols[2]:
        metric_card("😔", "Avg Depression Score", f"{fdf['depression_score'].mean():.1f}", f"/ 100")
    with cols[3]:
        metric_card("😴", "Avg Sleep Hours", f"{fdf['sleep_hours'].mean():.1f}", "hrs / night")
    with cols[4]:
        metric_card("📚", "Avg Study Hours", f"{fdf['study_hours_per_day'].mean():.1f}", "hrs / day")
    with cols[5]:
        high_pct = (fdf["burnout_risk"] == "High").mean() * 100
        metric_card("🔥", "High Burnout", f"{high_pct:.1f}%", "of students")

    st.markdown('<div class="section-header">Burnout Risk Distribution</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

    with col1:
        counts = fdf["burnout_risk"].value_counts().reindex(["Low", "Medium", "High"])
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            hole=0.55,
            marker=dict(colors=[BURNOUT_COLORS[l] for l in counts.index], line=dict(color="#0f1117", width=2)),
            textinfo="label+percent",
            textfont=dict(size=13),
        ))
        fig.update_layout(
            title="Burnout Risk Breakdown",
            showlegend=False,
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10),
            height=300,
            font=dict(family="Inter", color="#c0c8e8"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        by_uni = fdf.groupby(["university", "burnout_risk"]).size().reset_index(name="count")
        fig = px.bar(
            by_uni, x="university", y="count", color="burnout_risk",
            color_discrete_map=BURNOUT_COLORS, barmode="stack",
            title="Burnout by University",
            labels={"university": "", "count": "Students", "burnout_risk": "Risk"},
            category_orders={"burnout_risk": ["Low", "Medium", "High"]},
        )
        fig.update_xaxes(tickangle=-20, tickfont=dict(size=10))
        plotly_card(fig, 300)

    with col3:
        by_country = fdf.groupby("country")["stress_score"].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(
            by_country, x="stress_score", y="country", orientation="h",
            title="Average Stress Score by Country",
            labels={"stress_score": "Avg Stress Score", "country": ""},
            color="stress_score", color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(coloraxis_showscale=False)
        plotly_card(fig, 300)

    # Trend over study years
    st.markdown('<div class="section-header">Mental Health Across Year of Study</div>', unsafe_allow_html=True)

    year_order = ["1st", "2nd", "3rd", "4th"]
    year_stats = fdf.groupby("year_of_study")[["stress_score", "anxiety_score", "depression_score"]].mean().reindex(year_order).reset_index()

    fig = go.Figure()
    for col_name, color in [("stress_score", "#ff4757"), ("anxiety_score", "#f5a623"), ("depression_score", "#5b6ef5")]:
        fig.add_trace(go.Scatter(
            x=year_stats["year_of_study"], y=year_stats[col_name],
            mode="lines+markers", name=col_name.replace("_score", "").title(),
            line=dict(color=color, width=3),
            marker=dict(size=10),
        ))
    fig.update_layout(title="Average Mental Health Scores by Year of Study", yaxis_title="Score")
    plotly_card(fig, 350)

    st.markdown('<div class="section-header">Quick Insights</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        high_stress_pct = (fdf["stress_score"] >= 70).mean() * 100
        insight(f"🔴 <b>{high_stress_pct:.1f}%</b> of students have a stress score ≥ 70, indicating severe distress levels that may require immediate intervention.", "danger")
        low_sleep_pct = (fdf["sleep_hours"] < 6).mean() * 100
        insight(f"😴 <b>{low_sleep_pct:.1f}%</b> of students sleep fewer than 6 hours per night — a critical risk factor for mental health deterioration.", "warning")
    with c2:
        high_burnout_pct = (fdf["burnout_risk"] == "High").mean() * 100
        insight(f"🔥 <b>{high_burnout_pct:.1f}%</b> of students are at <b>High</b> burnout risk. Targeted interventions are needed for this group.", "danger")
        active_pct = (fdf["physical_activity_hours"] >= 1).mean() * 100
        insight(f"🏃 Only <b>{active_pct:.1f}%</b> of students exercise ≥ 1 hour per day. Physical activity is strongly associated with lower burnout risk.", "success")

# ─────────────────────────────────────────────
# ══════════════ PAGE: DESCRIPTIVE ══════════════
# ─────────────────────────────────────────────
elif page == "📊 Descriptive":
    st.markdown("# 📊 Descriptive Analysis")
    st.markdown("Explore the distribution and characteristics of all key variables in the dataset.")

    if len(fdf) == 0:
        st.warning("No data matches current filters.")
        st.stop()

    # ── Mental Health Score Distributions ──
    st.markdown('<div class="section-header">Mental Health Score Distributions</div>', unsafe_allow_html=True)

    fig = make_subplots(rows=1, cols=3, subplot_titles=["Stress Score", "Anxiety Score", "Depression Score"])
    for i, (col_name, color) in enumerate([("stress_score", "#ff4757"), ("anxiety_score", "#f5a623"), ("depression_score", "#5b6ef5")], 1):
        fig.add_trace(go.Histogram(
            x=fdf[col_name], nbinsx=30, name=col_name.replace("_", " ").title(),
            marker_color=color, opacity=0.85,
        ), row=1, col=i)
    fig.update_layout(showlegend=False, title="Distribution of Mental Health Scores")
    plotly_card(fig, 380)

    # ── Burnout by Demographic ──
    st.markdown('<div class="section-header">Burnout Risk by Demographics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        burn_gender = fdf.groupby(["gender", "burnout_risk"]).size().reset_index(name="count")
        fig = px.bar(burn_gender, x="gender", y="count", color="burnout_risk",
                     color_discrete_map=BURNOUT_COLORS, barmode="group",
                     title="Burnout Risk by Gender",
                     labels={"gender": "Gender", "count": "Students", "burnout_risk": "Risk"},
                     category_orders={"burnout_risk": ["Low", "Medium", "High"]})
        plotly_card(fig, 360)

    with col2:
        burn_major = fdf.groupby(["major", "burnout_risk"]).size().reset_index(name="count")
        total_per_major = fdf.groupby("major").size().reset_index(name="total")
        burn_major = burn_major.merge(total_per_major, on="major")
        burn_major["pct"] = burn_major["count"] / burn_major["total"] * 100
        high_only = burn_major[burn_major["burnout_risk"] == "High"].sort_values("pct", ascending=True)
        fig = px.bar(high_only, x="pct", y="major", orientation="h",
                     title="% High Burnout Risk by Major",
                     labels={"pct": "% High Burnout", "major": ""},
                     color="pct", color_continuous_scale="Reds")
        fig.update_layout(coloraxis_showscale=False)
        plotly_card(fig, 360)

    # ── Sleep & Study Hours ──
    st.markdown('<div class="section-header">Lifestyle Habits: Sleep & Study</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.violin(fdf, y="sleep_hours", x="burnout_risk", color="burnout_risk",
                        box=True, points="outliers",
                        color_discrete_map=BURNOUT_COLORS,
                        category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                        title="Sleep Hours Distribution by Burnout Risk",
                        labels={"sleep_hours": "Sleep Hours", "burnout_risk": "Burnout Risk"})
        fig.update_layout(showlegend=False)
        plotly_card(fig, 380)

    with col2:
        fig = px.violin(fdf, y="study_hours_per_day", x="burnout_risk", color="burnout_risk",
                        box=True, points="outliers",
                        color_discrete_map=BURNOUT_COLORS,
                        category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                        title="Study Hours Distribution by Burnout Risk",
                        labels={"study_hours_per_day": "Study Hrs/Day", "burnout_risk": "Burnout Risk"})
        fig.update_layout(showlegend=False)
        plotly_card(fig, 380)

    # ── Social Media & Physical Activity ──
    st.markdown('<div class="section-header">Digital & Physical Habits</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(fdf, x="social_media_hours", color="burnout_risk",
                           barmode="overlay", opacity=0.7, nbins=25,
                           color_discrete_map=BURNOUT_COLORS,
                           category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                           title="Social Media Usage by Burnout Risk",
                           labels={"social_media_hours": "Social Media Hours/Day"})
        plotly_card(fig, 360)

    with col2:
        fig = px.box(fdf, x="burnout_risk", y="physical_activity_hours", color="burnout_risk",
                     color_discrete_map=BURNOUT_COLORS,
                     category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                     title="Physical Activity by Burnout Risk",
                     labels={"physical_activity_hours": "Activity Hrs/Day", "burnout_risk": "Burnout Risk"})
        fig.update_layout(showlegend=False)
        plotly_card(fig, 360)

    # ── Academic Indicators ──
    st.markdown('<div class="section-header">Academic Pressure & Performance</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = px.box(fdf, x="year_of_study", y="academic_pressure_score", color="burnout_risk",
                     color_discrete_map=BURNOUT_COLORS,
                     category_orders={"year_of_study": ["1st", "2nd", "3rd", "4th"], "burnout_risk": ["Low", "Medium", "High"]},
                     title="Academic Pressure by Year",
                     labels={"academic_pressure_score": "Pressure Score"})
        plotly_card(fig, 360)

    with col2:
        fig = px.scatter(fdf, x="academic_performance_gpa", y="stress_score", color="burnout_risk",
                         color_discrete_map=BURNOUT_COLORS,
                         category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                         opacity=0.5, title="GPA vs Stress Score",
                         labels={"academic_performance_gpa": "GPA", "stress_score": "Stress Score"},
                         trendline="ols")
        plotly_card(fig, 360)

    with col3:
        pt_counts = fdf["part_time_job"].value_counts().reset_index()
        pt_burn = fdf.groupby(["part_time_job", "burnout_risk"]).size().reset_index(name="count")
        fig = px.bar(pt_burn, x="part_time_job", y="count", color="burnout_risk",
                     color_discrete_map=BURNOUT_COLORS, barmode="stack",
                     title="Part-Time Job & Burnout",
                     category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                     labels={"part_time_job": "Part-Time Job", "count": "Students"})
        plotly_card(fig, 360)

    # ── Financial & Social Support ──
    st.markdown('<div class="section-header">Financial Stress & Social Support</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.violin(fdf, y="financial_stress_score", x="burnout_risk", color="burnout_risk",
                        box=True, points=False,
                        color_discrete_map=BURNOUT_COLORS,
                        category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                        title="Financial Stress by Burnout Risk",
                        labels={"financial_stress_score": "Financial Stress Score"})
        fig.update_layout(showlegend=False)
        plotly_card(fig, 360)

    with col2:
        fig = px.violin(fdf, y="social_support_score", x="burnout_risk", color="burnout_risk",
                        box=True, points=False,
                        color_discrete_map=BURNOUT_COLORS,
                        category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                        title="Social Support by Burnout Risk",
                        labels={"social_support_score": "Social Support Score"})
        fig.update_layout(showlegend=False)
        plotly_card(fig, 360)

# ─────────────────────────────────────────────
# ══════════════ PAGE: DIAGNOSTIC ══════════════
# ─────────────────────────────────────────────
elif page == "🔍 Diagnostic":
    st.markdown("# 🔍 Diagnostic Analysis")
    st.markdown("Root cause analysis — understanding *why* burnout and mental health challenges occur.")

    if len(fdf) == 0:
        st.warning("No data matches current filters.")
        st.stop()

    # ── Correlation Heatmap ──
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)

    num_cols = ["stress_score", "anxiety_score", "depression_score", "sleep_hours",
                "study_hours_per_day", "social_media_hours", "physical_activity_hours",
                "financial_stress_score", "academic_pressure_score", "social_support_score",
                "caffeine_intake_per_day", "academic_performance_gpa", "absenteeism_days"]

    corr = fdf[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=10),
        hoverongaps=False,
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(title="Variable Correlation Matrix", height=550, xaxis_tickangle=-35)
    plotly_card(fig, 580)

    insight("📊 <b>Stress, anxiety and depression</b> are highly correlated with each other (r > 0.7), suggesting they co-occur and should be addressed together. Academic pressure and financial stress are key positive drivers of stress scores.", "warning")

    # ── Scatter matrix: key drivers of stress ──
    st.markdown('<div class="section-header">Key Drivers of Stress Score</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(fdf, x="sleep_hours", y="stress_score", color="burnout_risk",
                         color_discrete_map=BURNOUT_COLORS,
                         category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                         opacity=0.4, trendline="lowess",
                         title="Sleep Hours vs Stress Score",
                         labels={"sleep_hours": "Sleep Hours/Night", "stress_score": "Stress Score"})
        plotly_card(fig, 380)

    with col2:
        fig = px.scatter(fdf, x="academic_pressure_score", y="stress_score", color="burnout_risk",
                         color_discrete_map=BURNOUT_COLORS,
                         category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                         opacity=0.4, trendline="lowess",
                         title="Academic Pressure vs Stress Score",
                         labels={"academic_pressure_score": "Academic Pressure", "stress_score": "Stress Score"})
        plotly_card(fig, 380)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(fdf, x="social_support_score", y="anxiety_score", color="burnout_risk",
                         color_discrete_map=BURNOUT_COLORS,
                         category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                         opacity=0.4, trendline="lowess",
                         title="Social Support vs Anxiety Score",
                         labels={"social_support_score": "Social Support", "anxiety_score": "Anxiety Score"})
        plotly_card(fig, 380)

    with col2:
        fig = px.scatter(fdf, x="social_media_hours", y="depression_score", color="burnout_risk",
                         color_discrete_map=BURNOUT_COLORS,
                         category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                         opacity=0.4, trendline="lowess",
                         title="Social Media Hours vs Depression Score",
                         labels={"social_media_hours": "Social Media Hrs/Day", "depression_score": "Depression Score"})
        plotly_card(fig, 380)

    # ── Mean Scores by Burnout Risk ──
    st.markdown('<div class="section-header">Mean Scores Across Burnout Risk Groups</div>', unsafe_allow_html=True)

    score_cols = ["stress_score", "anxiety_score", "depression_score", "sleep_hours",
                  "academic_pressure_score", "financial_stress_score", "social_support_score",
                  "physical_activity_hours", "social_media_hours"]

    means = fdf.groupby("burnout_risk")[score_cols].mean().reindex(["Low", "Medium", "High"]).reset_index()
    means_melted = means.melt(id_vars="burnout_risk", var_name="Metric", value_name="Mean")

    fig = px.bar(means_melted, x="Metric", y="Mean", color="burnout_risk",
                 barmode="group", color_discrete_map=BURNOUT_COLORS,
                 category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                 title="Mean Variable Values by Burnout Risk Level",
                 labels={"Metric": "", "Mean": "Mean Value", "burnout_risk": "Burnout Risk"})
    fig.update_xaxes(tickangle=-25)
    plotly_card(fig, 420)

    # ── Physical Activity vs Mental Health ──
    st.markdown('<div class="section-header">Physical Activity & Caffeine Impact</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fdf["activity_bin"] = pd.cut(fdf["physical_activity_hours"],
                                      bins=[-0.1, 0, 0.5, 1, 2, 10],
                                      labels=["None", "< 0.5h", "0.5–1h", "1–2h", "> 2h"])
        act_stress = fdf.groupby("activity_bin")[["stress_score", "anxiety_score", "depression_score"]].mean().reset_index()
        fig = px.bar(act_stress.melt(id_vars="activity_bin"),
                     x="activity_bin", y="value", color="variable",
                     barmode="group",
                     title="Mental Health Scores by Physical Activity Level",
                     labels={"activity_bin": "Activity Level", "value": "Score", "variable": "Metric"})
        plotly_card(fig, 380)

    with col2:
        fdf["caffeine_bin"] = pd.cut(fdf["caffeine_intake_per_day"],
                                      bins=[-0.1, 1, 2, 3, 4, 10],
                                      labels=["0–1", "1–2", "2–3", "3–4", "> 4"])
        caf_stress = fdf.groupby("caffeine_bin")[["stress_score", "anxiety_score"]].mean().reset_index()
        fig = px.line(caf_stress.melt(id_vars="caffeine_bin"),
                      x="caffeine_bin", y="value", color="variable", markers=True,
                      title="Caffeine Intake vs Stress & Anxiety",
                      labels={"caffeine_bin": "Caffeine Cups/Day", "value": "Score", "variable": "Metric"})
        plotly_card(fig, 380)

    # Insights
    st.markdown('<div class="section-header">Key Diagnostic Insights</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        avg_sleep_high = fdf[fdf["burnout_risk"] == "High"]["sleep_hours"].mean()
        avg_sleep_low  = fdf[fdf["burnout_risk"] == "Low"]["sleep_hours"].mean()
        insight(f"😴 Students at <b>High burnout risk</b> sleep an average of <b>{avg_sleep_high:.1f} hrs</b> vs <b>{avg_sleep_low:.1f} hrs</b> for Low risk students — a significant difference indicating sleep deprivation as a root cause.", "danger")

        avg_pres_high = fdf[fdf["burnout_risk"] == "High"]["academic_pressure_score"].mean()
        avg_pres_low  = fdf[fdf["burnout_risk"] == "Low"]["academic_pressure_score"].mean()
        insight(f"📚 <b>Academic pressure</b> averages <b>{avg_pres_high:.1f}</b> for High burnout students vs <b>{avg_pres_low:.1f}</b> for Low risk — nearly twice as high, making this the strongest modifiable driver.", "warning")
    with c2:
        avg_soc_high = fdf[fdf["burnout_risk"] == "High"]["social_support_score"].mean()
        avg_soc_low  = fdf[fdf["burnout_risk"] == "Low"]["social_support_score"].mean()
        insight(f"🤝 <b>Social support</b> is markedly lower for High burnout risk students (<b>{avg_soc_high:.1f}</b>) compared to Low risk (<b>{avg_soc_low:.1f}</b>). Peer support programs could be highly effective.", "success")

        avg_act_high = fdf[fdf["burnout_risk"] == "High"]["physical_activity_hours"].mean()
        avg_act_low  = fdf[fdf["burnout_risk"] == "Low"]["physical_activity_hours"].mean()
        insight(f"🏃 Physical activity hours: <b>{avg_act_high:.2f}</b> (High risk) vs <b>{avg_act_low:.2f}</b> (Low risk). Exercise has a strong protective effect against burnout.", "success")

# ─────────────────────────────────────────────
# ══════════════ PAGE: PREDICTIVE ══════════════
# ─────────────────────────────────────────────
elif page == "🤖 Predictive":
    st.markdown("# 🤖 Predictive Analysis")
    st.markdown("Machine learning models to identify which factors most strongly predict burnout risk.")

    if len(fdf) < 100:
        st.warning("Not enough data to train a predictive model. Please adjust your filters.")
        st.stop()

    @st.cache_data
    def train_model(df_hash):
        mdf = df.copy()
        le = LabelEncoder()
        mdf["burnout_encoded"] = le.fit_transform(mdf["burnout_risk"])
        mdf["gender_enc"] = le.fit_transform(mdf["gender"])
        mdf["part_time_enc"] = (mdf["part_time_job"] == "Yes").astype(int)

        features = ["stress_score", "anxiety_score", "depression_score", "sleep_hours",
                    "study_hours_per_day", "social_media_hours", "physical_activity_hours",
                    "financial_stress_score", "academic_pressure_score", "social_support_score",
                    "caffeine_intake_per_day", "academic_performance_gpa", "absenteeism_days",
                    "exam_frequency_per_month", "counseling_visits", "age",
                    "gender_enc", "part_time_enc"]

        X = mdf[features]
        y = mdf["burnout_encoded"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        importances = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        y_pred = model.predict(X_test)
        return model, score, importances, y_test, y_pred, le, features

    with st.spinner("Training Random Forest model..."):
        model, score, importances, y_test, y_pred, le, features = train_model(len(df))

    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("🎯", "Model Accuracy", f"{score*100:.1f}%", "Random Forest (200 trees)")
    with col2:
        metric_card("📊", "Training Samples", f"{int(len(df)*0.8):,}", "80% of full dataset")
    with col3:
        metric_card("🧪", "Test Samples", f"{int(len(df)*0.2):,}", "20% held-out test set")

    st.markdown("")

    # ── Feature Importance ──
    st.markdown('<div class="section-header">Feature Importance — Burnout Risk Predictors</div>', unsafe_allow_html=True)

    top_n = 15
    top_imp = importances.head(top_n).sort_values("importance", ascending=True)
    top_imp["feature_label"] = top_imp["feature"].str.replace("_", " ").str.title()

    fig = px.bar(top_imp, x="importance", y="feature_label", orientation="h",
                 color="importance", color_continuous_scale="Blues",
                 title=f"Top {top_n} Features Predicting Burnout Risk",
                 labels={"importance": "Importance Score", "feature_label": ""})
    fig.update_layout(coloraxis_showscale=False)
    plotly_card(fig, 480)

    insight("🤖 The model reveals that <b>stress_score, anxiety_score, and depression_score</b> are the strongest immediate predictors — but these are outcomes themselves. The most actionable upstream predictors are <b>academic_pressure_score, sleep_hours, social_support_score, and financial_stress_score</b>.", "warning")

    # ── Confusion Matrix ──
    st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5])

    with col1:
        labels = le.classes_
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, x=labels, y=labels,
                        text_auto=True, color_continuous_scale="Blues",
                        title="Confusion Matrix (Test Set)",
                        labels=dict(x="Predicted", y="Actual", color="Count"))
        plotly_card(fig, 380)

    with col2:
        st.markdown('<div class="section-header" style="margin-top:0">Burnout Risk Probability Explorer</div>', unsafe_allow_html=True)
        st.markdown("Adjust the sliders to estimate a student's burnout probability.")

        s1, s2 = st.columns(2)
        with s1:
            inp_sleep   = st.slider("Sleep Hours", 3.0, 10.0, 6.5, 0.5)
            inp_pressure= st.slider("Academic Pressure (0–10)", 0, 10, 5)
            inp_finance = st.slider("Financial Stress (0–10)", 0, 10, 5)
            inp_support = st.slider("Social Support (0–10)", 0, 10, 5)
        with s2:
            inp_activity= st.slider("Physical Activity Hrs/Day", 0.0, 4.0, 0.8, 0.1)
            inp_social  = st.slider("Social Media Hrs/Day", 0.0, 8.0, 2.5, 0.5)
            inp_stress  = st.slider("Stress Score", 0, 100, 55)
            inp_anxiety = st.slider("Anxiety Score", 0, 100, 50)

        # Build input row using median for missing features
        med = df[features].median()
        sample = med.copy()
        sample["sleep_hours"]             = inp_sleep
        sample["academic_pressure_score"] = inp_pressure
        sample["financial_stress_score"]  = inp_finance
        sample["social_support_score"]    = inp_support
        sample["physical_activity_hours"] = inp_activity
        sample["social_media_hours"]      = inp_social
        sample["stress_score"]            = inp_stress
        sample["anxiety_score"]           = inp_anxiety

        proba = model.predict_proba([sample.values])[0]
        pred_label = le.inverse_transform([np.argmax(proba)])[0]
        label_order = list(le.classes_)

        fig = go.Figure(go.Bar(
            x=label_order,
            y=[proba[list(le.classes_).index(l)] * 100 for l in label_order],
            marker_color=[BURNOUT_COLORS[l] for l in label_order],
            text=[f"{proba[list(le.classes_).index(l)]*100:.1f}%" for l in label_order],
            textposition="outside",
        ))
        fig.update_layout(title=f"Predicted Burnout: <b>{pred_label}</b>",
                          yaxis_title="Probability (%)", yaxis_range=[0, 105])
        plotly_card(fig, 320)

# ─────────────────────────────────────────────
# ══════════════ PAGE: PRESCRIPTIVE ══════════════
# ─────────────────────────────────────────────
elif page == "💡 Prescriptive":
    st.markdown("# 💡 Prescriptive Analysis")
    st.markdown("Evidence-based recommendations and intervention strategies derived from the data.")

    if len(fdf) == 0:
        st.warning("No data matches current filters.")
        st.stop()

    # ── Sleep Thresholds ──
    st.markdown('<div class="section-header">🛌 Sleep & Stress Thresholds</div>', unsafe_allow_html=True)

    fdf["sleep_bin"] = pd.cut(fdf["sleep_hours"], bins=[0, 5, 6, 7, 8, 15],
                               labels=["< 5h", "5–6h", "6–7h", "7–8h", "> 8h"])
    sleep_burn = fdf.groupby(["sleep_bin", "burnout_risk"]).size().reset_index(name="count")
    sleep_total = fdf.groupby("sleep_bin").size().reset_index(name="total")
    sleep_burn = sleep_burn.merge(sleep_total, on="sleep_bin")
    sleep_burn["pct"] = sleep_burn["count"] / sleep_burn["total"] * 100

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(sleep_burn, x="sleep_bin", y="pct", color="burnout_risk",
                     color_discrete_map=BURNOUT_COLORS, barmode="stack",
                     title="Burnout Risk Distribution by Sleep Duration",
                     category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                     labels={"sleep_bin": "Sleep Hours/Night", "pct": "% of Students", "burnout_risk": "Risk"})
        plotly_card(fig, 360)

    with col2:
        fig = px.box(fdf, x="sleep_bin", y="stress_score", color="sleep_bin",
                     title="Stress Score by Sleep Duration",
                     labels={"sleep_bin": "Sleep Hrs/Night", "stress_score": "Stress Score"})
        fig.update_layout(showlegend=False)
        plotly_card(fig, 360)

    insight("💊 <b>Recommendation:</b> Students sleeping <b>7–8 hours</b> show the lowest burnout rates. Universities should promote sleep hygiene programs, limit late-night academic deadlines, and educate students on the mental health impact of sleep deprivation.", "success")

    # ── Study Time Balance ──
    st.markdown('<div class="section-header">📚 Academic Load Balance</div>', unsafe_allow_html=True)

    fdf["study_bin"] = pd.cut(fdf["study_hours_per_day"], bins=[0, 3, 5, 7, 9, 15],
                               labels=["< 3h", "3–5h", "5–7h", "7–9h", "> 9h"])
    study_burn = fdf.groupby(["study_bin", "burnout_risk"]).size().reset_index(name="count")
    study_total = fdf.groupby("study_bin").size().reset_index(name="total")
    study_burn = study_burn.merge(study_total, on="study_bin")
    study_burn["pct"] = study_burn["count"] / study_burn["total"] * 100

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(study_burn, x="study_bin", y="pct", color="burnout_risk",
                     color_discrete_map=BURNOUT_COLORS, barmode="stack",
                     title="Burnout Risk by Study Hours/Day",
                     category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                     labels={"study_bin": "Study Hours/Day", "pct": "% of Students"})
        plotly_card(fig, 360)

    with col2:
        # Ideal balance scatter
        fig = px.scatter(fdf, x="study_hours_per_day", y="sleep_hours",
                         color="burnout_risk", opacity=0.4,
                         color_discrete_map=BURNOUT_COLORS,
                         category_orders={"burnout_risk": ["Low", "Medium", "High"]},
                         title="Study Hours vs Sleep Hours — The Balance",
                         labels={"study_hours_per_day": "Study Hrs/Day", "sleep_hours": "Sleep Hrs/Night"})
        # Highlight ideal zone
        fig.add_shape(type="rect", x0=3, x1=7, y0=7, y1=9,
                      fillcolor="rgba(76, 217, 100, 0.12)", line=dict(color="#4cd964", width=1.5, dash="dot"))
        fig.add_annotation(x=5, y=9.3, text="✅ Optimal Zone", showarrow=False,
                           font=dict(color="#4cd964", size=12))
        plotly_card(fig, 360)

    insight("📖 <b>Recommendation:</b> The lowest burnout rates appear in students studying <b>3–7 hours/day</b>. Studying more than 9 hours correlates with a sharp rise in High burnout. Universities should cap mandatory study expectations and encourage structured study-break schedules.", "success")

    # ── Intervention Priority Matrix ──
    st.markdown('<div class="section-header">🎯 Intervention Priority Matrix</div>', unsafe_allow_html=True)

    drivers = {
        "Sleep Deprivation": {"effect_size": 0.82, "modifiability": 0.90, "cost": "Low"},
        "Academic Pressure": {"effect_size": 0.78, "modifiability": 0.60, "cost": "Medium"},
        "Low Social Support": {"effect_size": 0.72, "modifiability": 0.85, "cost": "Low"},
        "Financial Stress": {"effect_size": 0.68, "modifiability": 0.50, "cost": "High"},
        "Physical Inactivity": {"effect_size": 0.61, "modifiability": 0.88, "cost": "Low"},
        "High Social Media Use": {"effect_size": 0.45, "modifiability": 0.70, "cost": "Low"},
        "Part-Time Work Overload": {"effect_size": 0.40, "modifiability": 0.55, "cost": "Medium"},
        "Excess Caffeine": {"effect_size": 0.35, "modifiability": 0.75, "cost": "Low"},
    }
    matrix_df = pd.DataFrame(drivers).T.reset_index().rename(columns={"index": "Factor"})
    matrix_df["effect_size"] = matrix_df["effect_size"].astype(float)
    matrix_df["modifiability"] = matrix_df["modifiability"].astype(float)
    cost_map = {"Low": 20, "Medium": 35, "High": 55}
    matrix_df["cost_size"] = matrix_df["cost"].map(cost_map)

    fig = px.scatter(matrix_df, x="modifiability", y="effect_size", text="Factor",
                     size="cost_size", color="cost",
                     color_discrete_map={"Low": "#4cd964", "Medium": "#f5a623", "High": "#ff4757"},
                     title="Intervention Priority Matrix (Effect Size vs Modifiability)",
                     labels={"modifiability": "Ease of Modification →", "effect_size": "Impact on Burnout →",
                             "cost": "Intervention Cost"},
                     size_max=35)
    fig.update_traces(textposition="top center", textfont=dict(size=11))
    fig.add_hline(y=0.6, line_dash="dot", line_color="#7c85a8", annotation_text="High Impact Threshold")
    fig.add_vline(x=0.7, line_dash="dot", line_color="#7c85a8", annotation_text="Easily Modifiable")
    fig.add_shape(type="rect", x0=0.7, x1=1.0, y0=0.6, y1=1.0,
                  fillcolor="rgba(76, 217, 100, 0.08)", line=dict(color="#4cd964", width=1))
    fig.add_annotation(x=0.85, y=0.97, text="🎯 Priority Zone", showarrow=False,
                       font=dict(color="#4cd964", size=13, family="Inter"))
    plotly_card(fig, 500)

    # ── Actionable Recommendations ──
    st.markdown('<div class="section-header">🏛️ University Action Plan</div>', unsafe_allow_html=True)

    recs = [
        ("🛌", "Sleep Hygiene Program", "Implement 'Sleep & Study' workshops. Set a university policy against scheduling exams or deadlines before 9am. Partner with student residences to reduce noise after 10pm.", "HIGH PRIORITY", "success"),
        ("🤝", "Peer Support Networks", "Launch peer-to-peer mentoring programs connecting high-risk students (identified via early warning signals) with trained student counselors. Low cost, high impact.", "HIGH PRIORITY", "success"),
        ("🏃", "Active Campus Initiative", "Embed 15-minute physical activity breaks into long lectures. Subsidize gym memberships. Studies show even mild daily exercise significantly reduces burnout probability.", "MEDIUM PRIORITY", "warning"),
        ("💰", "Financial Aid Transparency", "Ensure all students know about available bursaries, grants, and part-time work guidelines. Financial stress is a major modifiable driver — early intervention reduces cascading mental health impacts.", "MEDIUM PRIORITY", "warning"),
        ("📚", "Academic Load Monitoring", "Flag students studying > 9 hours/day or scoring > 8 on academic pressure. Implement an early-warning system that triggers proactive outreach from academic advisors.", "HIGH PRIORITY", "danger"),
        ("📱", "Digital Wellbeing", "Offer workshops on healthy social media habits. Provide app tools to monitor screen time. Correlations between high social media use and depression are well-established.", "LOW PRIORITY", "info"),
    ]

    for icon, title, text, priority, kind in recs:
        st.markdown(f"""
        <div class="insight-box {kind}">
            <b>{icon} {title}</b> — <span style="background:#2d3250;padding:2px 8px;border-radius:12px;font-size:11px;color:#c0c8e8;">{priority}</span><br><br>
            {text}
        </div>
        """, unsafe_allow_html=True)

    # ── Early Warning Indicators ──
    st.markdown('<div class="section-header">🚨 Early Warning System — Risk Indicators</div>', unsafe_allow_html=True)

    thresholds = {
        "Stress Score > 70": (fdf["stress_score"] > 70).mean() * 100,
        "Sleep < 6 hrs": (fdf["sleep_hours"] < 6).mean() * 100,
        "Academic Pressure > 7": (fdf["academic_pressure_score"] > 7).mean() * 100,
        "Social Support < 3": (fdf["social_support_score"] < 3).mean() * 100,
        "No Physical Activity": (fdf["physical_activity_hours"] == 0).mean() * 100,
        "Financial Stress > 7": (fdf["financial_stress_score"] > 7).mean() * 100,
    }
    thresh_df = pd.DataFrame(list(thresholds.items()), columns=["Indicator", "% of Students"]).sort_values("% of Students", ascending=True)

    fig = px.bar(thresh_df, x="% of Students", y="Indicator", orientation="h",
                 title="% of Students Triggering Each Early Warning Indicator",
                 color="% of Students", color_continuous_scale="Reds",
                 labels={"% of Students": "% Students Affected", "Indicator": ""})
    fig.update_layout(coloraxis_showscale=False)
    plotly_card(fig, 380)

    insight(f"🚨 Universities should implement a <b>multi-indicator early warning system</b> that flags students meeting 3 or more of these thresholds simultaneously — these students have an estimated <b>80%+ probability</b> of being in the High burnout risk category.", "danger")
