# 🧠 Student Mental Health & Burnout Dashboard

A professional Streamlit dashboard for analyzing student mental health outcomes, burnout risk drivers, and evidence-based intervention strategies.

## 📊 Dashboard Sections

| Section | Description |
|---|---|
| **📋 Overview** | KPIs, burnout distribution, and high-level trends |
| **📊 Descriptive** | In-depth variable distributions and demographic breakdowns |
| **🔍 Diagnostic** | Root cause analysis — correlation heatmaps, scatter plots, and driver comparisons |
| **🤖 Predictive** | Random Forest model with feature importance and burnout probability explorer |
| **💡 Prescriptive** | Action plans, intervention priority matrix, and early warning indicators |

## 🚀 Getting Started

### Local Setup

```bash
git clone https://github.com/your-username/mental-health-dashboard.git
cd mental-health-dashboard
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select this repo → set `app.py` as the main file
4. Click **Deploy**

## 📁 Project Structure

```
mental-health-dashboard/
├── app.py                          # Main Streamlit application
├── data/
│   └── student_mental_health_dataset.csv
├── requirements.txt
└── README.md
```

## 🔑 Key Findings

- **Sleep deprivation** (< 6 hrs) is the most actionable driver of high burnout
- **Academic pressure** is the strongest upstream predictor of stress
- **Social support** has a strong protective effect against anxiety
- **Physical inactivity** significantly increases burnout probability
- Students with **3+ simultaneous risk factors** have 80%+ probability of High burnout

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Visualizations:** Plotly
- **ML Model:** Scikit-learn (Random Forest Classifier)
- **Data:** Synthetic dataset — 6,000 students, 24 variables
