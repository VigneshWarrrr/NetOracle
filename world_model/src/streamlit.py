"""
NetOracle - Real Network Attack Forecasting Dashboard

Fully offline Streamlit interface.

Real CIC-IDS-2018 CSV
        ->
10-second network states
        ->
Temporal World Model
        ->
K-step future simulation
        ->
Attack probability + MITRE stage + explanation
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )

from explainability.graph_explainer import GraphExplainer
from explainability.shap_explainer import WorldModelExplainer
from forecasting.attack_chain import AttackChainBuilder
from forecasting.mitre_mapping import MitreMapper as MITREMapper
from forecasting.victim import NextVictimForecaster
from world_model.inference import NetOracleInference


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NetOracle",
    page_icon="🛡️",
    layout="wide",
)


# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
        }

        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
        }

        .subtitle {
            color: #9ca3af;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .critical-box {
            border-left: 5px solid #ef4444;
            padding: 15px;
            background-color: rgba(239,68,68,0.08);
            border-radius: 8px;
        }

        .safe-box {
            border-left: 5px solid #22c55e;
            padding: 15px;
            background-color: rgba(34,197,94,0.08);
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL ONCE
# =========================================================

@st.cache_resource
def load_engine():

    return NetOracleInference(
        model_path=
            "models/best_world_model.pth",

        config_path=
            "configs/config.yaml",
    )


try:

    engine = load_engine()

    model_ready = True

except Exception as error:

    model_ready = False

    st.error(
        f"Failed to load model: {error}"
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ NetOracle'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    World Model based predictive cyber defence —
    forecasting network attack progression before compromise.
    </div>
    """,
    unsafe_allow_html=True,
)


if model_ready:

    device_name = str(
        engine.device
    )

    st.caption(
        f"Model status: Loaded | "
        f"Device: {device_name.upper()} | "
        f"Forecast horizon: 50 seconds"
    )


# =========================================================
# TABS
# =========================================================

tabs = st.tabs(
    [
        "🔮 Real Traffic Forecast",
        "📊 Benchmark",
        "🔍 Explainability",
        "🗺️ MITRE ATT&CK",
        "🧠 Architecture",
    ]
)


# =========================================================
# TAB 1 — REAL TRAFFIC
# =========================================================

with tabs[0]:

    st.subheader(
        "Real CIC-IDS-2018 Traffic Analysis"
    )

    st.write(
        "Upload a CIC-IDS-2018 flow CSV. "
        "NetOracle converts it into real "
        "10-second network states and performs "
        "recursive future simulation."
    )

    uploaded_file = st.file_uploader(
        "Upload CIC-IDS-2018 CSV",
        type=["csv"],
    )

    demo_mode = st.checkbox(
        "Use local held-out infiltration demo",
        value=False,
    )

    stride = st.slider(
        "Timeline prediction interval",
        min_value=1,
        max_value=20,
        value=5,
        help=(
            "5 = generate one forecast "
            "every 50 seconds."
        ),
    )

    run_analysis = st.button(
        "🚀 Analyze Traffic",
        type="primary",
    )

    if run_analysis:

        if not model_ready:

            st.error(
                "Model is not available."
            )

        elif (
            uploaded_file is None
            and not demo_mode
        ):

            st.warning(
                "Upload a CSV or enable "
                "the held-out demo."
            )

        else:

            temp_path = None

            try:

                # -----------------------------------------
                # Choose data source
                # -----------------------------------------

                if demo_mode:

                    csv_path = (
                        "data/raw/"
                        "Wednesday-28-02-2018_"
                        "TrafficForML_CICFlowMeter.csv"
                    )

                    start_time = (
                        "2018-02-28 10:00:00"
                    )

                    end_time = (
                        "2018-02-28 13:00:00"
                    )

                    source_name = (
                        "Held-out Wednesday-28 "
                        "late infiltration episode"
                    )

                else:

                    suffix = ".csv"

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix,
                    ) as temporary_file:

                        temporary_file.write(
                            uploaded_file.getbuffer()
                        )

                        temp_path = (
                            temporary_file.name
                        )

                    csv_path = temp_path
                    start_time = None
                    end_time = None

                    source_name = (
                        uploaded_file.name
                    )

                # -----------------------------------------
                # Run REAL model
                # -----------------------------------------

                with st.spinner(
                    "Building network states and "
                    "running temporal world model..."
                ):

                    timeline = (
                        engine.predict_timeline(
                            csv_path,
                            start_time=
                                start_time,

                            end_time=
                                end_time,

                            stride=
                                stride,
                        )
                    )

                risks = (
                    timeline[
                        "risk_timeline"
                    ]
                )

                stages = (
                    timeline[
                        "stage_timeline"
                    ]
                )

                threshold = float(
                    timeline[
                        "threshold"
                    ]
                )

                observed = (
                    timeline[
                        "observed_timeline"
                    ]
                )

                # -----------------------------------------
                # Summary metrics
                # -----------------------------------------

                max_risk = float(
                    np.max(risks)
                )

                mean_risk = float(
                    np.mean(risks)
                )

                alert_count = int(
                    np.sum(
                        risks >= threshold
                    )
                )

                dominant_stage = int(
                    stages[
                        np.argmax(risks)
                    ]
                )

                stage_info = (
                    MITREMapper
                    .get_stage_info(
                        dominant_stage
                    )
                )

                st.success(
                    f"Analyzed: {source_name}"
                )

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                col1.metric(
                    "Maximum Forecast Risk",
                    f"{max_risk:.1%}",
                )

                col2.metric(
                    "Mean Forecast Risk",
                    f"{mean_risk:.1%}",
                )

                col3.metric(
                    "Forecast Alerts",
                    alert_count,
                )

                col4.metric(
                    "Highest-Risk Stage",
                    stage_info["name"],
                )

                # -----------------------------------------
                # Risk timeline
                # -----------------------------------------

                st.subheader(
                    "Forecast Risk Timeline"
                )

                x_axis = np.arange(
                    len(risks)
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=risks,
                        mode="lines",
                        name=
                            "Predicted future risk",

                        line=dict(
                            color="#ef4444",
                            width=2,
                        ),

                        fill="tozeroy",

                        fillcolor=
                            "rgba(239,68,68,0.12)",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=[
                            threshold
                        ] * len(risks),

                        mode="lines",

                        name=
                            "Alert threshold",

                        line=dict(
                            color="#f59e0b",
                            width=2,
                            dash="dash",
                        ),
                    )
                )

                # Observed ground-truth markers.
                attack_points = (
                    np.where(
                        observed == 1
                    )[0]
                )

                if len(
                    attack_points
                ) > 0:

                    fig.add_trace(
                        go.Scatter(
                            x=attack_points,

                            y=np.ones(
                                len(
                                    attack_points
                                )
                            ),

                            mode="markers",

                            name=
                                "Observed malicious state",

                            marker=dict(
                                color="#a855f7",
                                size=6,
                            ),
                        )
                    )

                fig.update_layout(
                    template="plotly_dark",
                    height=430,
                    xaxis_title=
                        "Chronological forecast step",

                    yaxis_title=
                        "Infiltration probability",

                    yaxis=dict(
                        range=[0, 1]
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                # -----------------------------------------
                # Stage timeline
                # -----------------------------------------

                st.subheader(
                    "Predicted MITRE Stage Timeline"
                )

                stage_names = [
                    MITREMapper
                    .get_stage_name(
                        int(stage)
                    )

                    for stage in stages
                ]

                stage_df = pd.DataFrame(
                    {
                        "Forecast":
                            x_axis,

                        "Risk":
                            risks,

                        "MITRE Stage":
                            stage_names,

                        "Alert":
                            risks
                            >= threshold,
                    }
                )

                st.dataframe(
                    stage_df,
                    use_container_width=True,
                    height=300,
                )

                # -----------------------------------------
                # Most dangerous point
                # -----------------------------------------

                peak_index = int(
                    np.argmax(risks)
                )

                peak_stage = int(
                    stages[
                        peak_index
                    ]
                )

                peak_info = (
                    MITREMapper
                    .get_stage_info(
                        peak_stage
                    )
                )

                if (
                    max_risk
                    >= threshold
                ):

                    st.markdown(
                        f"""
                        <div class="critical-box">

                        <h3>
                        ⚠️ Predictive Security Alert
                        </h3>

                        <b>
                        Predicted risk:
                        {max_risk:.1%}
                        </b>

                        <br>

                        Predicted stage:
                        <b>
                        {peak_info['name']}
                        </b>

                        <br>

                        MITRE tactic:
                        {peak_info['mitre_id']}

                        <br><br>

                        Recommended action:
                        {peak_info['recommended_action']}

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:

                    st.markdown(
                        """
                        <div class="safe-box">
                        <h3>
                        ✅ No Forecast Threshold Exceeded
                        </h3>
                        Continue monitoring.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as error:

                st.exception(error)

            finally:

                if (
                    temp_path is not None
                    and os.path.exists(
                        temp_path
                    )
                ):

                    os.remove(
                        temp_path
                    )


# =========================================================
# TAB 2 — BENCHMARK
# =========================================================

with tabs[1]:

    st.subheader(
        "Temporal World Model vs Static Baseline"
    )

    st.write(
        "Both models were evaluated on the "
        "same chronologically held-out traffic."
    )

    comparison = pd.DataFrame(
        {
            "Metric": [
                "F1 Score",
                "AUC-ROC",
                "Precision",
                "Recall",
                "False Positive Rate",
            ],

            "World Model": [
                0.6941,
                0.8323,
                0.6201,
                0.7882,
                0.3436,
            ],

            "Logistic Regression": [
                0.5834,
                0.6355,
                0.4863,
                0.7289,
                0.5478,
            ],
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="World Model",
            x=comparison["Metric"],
            y=comparison["World Model"],
            marker_color="#22c55e",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Logistic Regression",
            x=comparison["Metric"],
            y=comparison[
                "Logistic Regression"
            ],
            marker_color="#64748b",
        )
    )

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        height=430,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        "The temporal World Model improves "
        "T+1 F1 and AUC while substantially "
        "reducing false-positive rate compared "
        "with a static Logistic Regression model."
    )


# =========================================================
# TAB 3 — EXPLAINABILITY
# =========================================================

with tabs[2]:

    st.subheader(
        "Explainable Forecasting"
    )

    st.write(
        "NetOracle uses input-gradient attribution "
        "to estimate which network-state features "
        "most influence the predicted attack risk."
    )

    if model_ready:

        demo_file = (
            "data/raw/"
            "Wednesday-28-02-2018_"
            "TrafficForML_CICFlowMeter.csv"
        )

        if st.button(
            "Generate Real Traffic Explanation"
        ):

            try:

                with st.spinner(
                    "Calculating feature attribution..."
                ):

                    prepared = (
                        engine.prepare_csv(
                            demo_file,

                            start_time=
                                "2018-02-28 10:00:00",

                            end_time=
                                "2018-02-28 12:00:00",
                        )
                    )

                    explainer = (
                        WorldModelExplainer(
                            engine.model,
                            engine.feature_names,
                        )
                    )

                    features = (
                        explainer.get_top_features(
                            prepared[
                                "input_tensor"
                            ],
                            top_k=10,
                        )
                    )

                feature_names = [
                    item[0]
                    for item in features
                ]

                scores = [
                    item[1]
                    for item in features
                ]

                fig = go.Figure(
                    go.Bar(
                        x=scores,
                        y=feature_names,
                        orientation="h",

                        marker=dict(
                            color=scores,
                            colorscale="Reds",
                        ),
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=450,

                    title=
                        "Features Driving the Forecast",

                    xaxis_title=
                        "Attribution magnitude",

                    yaxis=dict(
                        autorange="reversed"
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.caption(
                    "Higher attribution indicates "
                    "greater influence on the "
                    "forecasted attack probability."
                )

            except Exception as error:

                st.exception(error)


# =========================================================
# TAB 4 — MITRE
# =========================================================

with tabs[3]:

    st.subheader(
        "MITRE ATT&CK Decision Support"
    )

    for stage_id in range(7):

        info = (
            MITREMapper
            .get_stage_info(
                stage_id
            )
        )

        with st.expander(
            f"{info['name']} "
            f"({info['mitre_id']})"
        ):

            st.write(
                info["description"]
            )

            st.write(
                "**Severity:**",
                f"{info['severity']}/10",
            )

            st.write(
                "**Recommended Action:**",
                info[
                    "recommended_action"
                ],
            )


# =========================================================
# TAB 5 — ARCHITECTURE
# =========================================================

with tabs[4]:

    st.subheader(
        "World Model Architecture"
    )

    st.code(
        """
Real CIC-IDS-2018 Flow Telemetry
             │
             ▼
     10-second Network States
             │
             ▼
        State Encoder
             │
             ▼
 Causal Temporal Transformer
             │
             ▼
       Latent State Z(t)
             │
             ▼
  Learned Transition Dynamics
             │
       ┌─────┴─────┐
       ▼           ▼
    Z(t+1)      Network State
       │
       ├────────► Attack Probability
       │
       └────────► MITRE Stage
       │
       ▼
    Z(t+2)
       │
      ...
       ▼
    Z(t+5)

Observed context : 200 seconds
Forecast horizon : 50 seconds
        """
    )

    st.markdown(
        """
        Key properties:

        - Recursive K-step future simulation
        - Causal Transformer attention
        - Training-only normalization
        - Chronological train/validation/test separation
        - MITRE ATT&CK decision support
        - Explainable feature attribution
        - Fully offline inference
        """
    )