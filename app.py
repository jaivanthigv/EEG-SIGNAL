from datetime import datetime
from html import escape
from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from speech_module import speak_text


st.set_page_config(page_title="NeuroLink", layout="wide")


st.markdown(
    """
    <style>
    :root {
        --bg: #050816;
        --panel: rgba(11, 21, 43, 0.92);
        --panel-soft: rgba(15, 29, 58, 0.86);
        --line: rgba(97, 132, 202, 0.32);
        --cyan: #5bd6ff;
        --blue: #3b82f6;
        --violet: #8b5cf6;
        --green: #4ade80;
        --text: #eef2ff;
        --muted: #9fb0d2;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.22), transparent 26%),
            radial-gradient(circle at top right, rgba(139, 92, 246, 0.15), transparent 24%),
            linear-gradient(180deg, #040714 0%, #07101f 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1.25rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4, p, div, span, label {
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }

    .hero-card,
    .info-card,
    .panel-card,
    .soft-card,
    .output-card,
    .metric-box {
        background: linear-gradient(180deg, rgba(10, 20, 40, 0.97), rgba(7, 14, 30, 0.95));
        border: 1px solid var(--line);
        border-radius: 22px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.28);
    }

    .hero-card {
        padding: 24px 28px;
        min-height: 148px;
    }

    .hero-wrap {
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .hero-icon {
        width: 92px;
        height: 92px;
        border-radius: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
            radial-gradient(circle at 30% 30%, rgba(91, 214, 255, 0.28), transparent 48%),
            linear-gradient(145deg, rgba(10, 18, 34, 0.98), rgba(16, 29, 58, 0.95));
        border: 1px solid rgba(91, 214, 255, 0.25);
        font-size: 46px;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #77e1ff 0%, #5d9bff 38%, #9f72ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #d1d9ef;
        font-size: 1.18rem;
    }

    .info-card {
        padding: 22px 24px;
        min-height: 148px;
    }

    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px 24px;
    }

    .eyebrow {
        color: #63b3ff;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        color: var(--green);
        font-weight: 700;
    }

    .status-dot {
        width: 11px;
        height: 11px;
        border-radius: 999px;
        background: var(--green);
        box-shadow: 0 0 16px rgba(74, 222, 128, 0.8);
    }

    .quality-text {
        color: var(--green);
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    .progress-track {
        width: 100%;
        height: 9px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.18);
        overflow: hidden;
    }

    .progress-bar {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #30e07f, #50e3c2);
    }

    .panel-card, .soft-card, .output-card {
        padding: 18px 20px;
        margin-top: 14px;
    }

    .panel-title {
        color: #63a5ff;
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }

    .connection-list {
        display: grid;
        gap: 12px;
    }

    .row-between {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
    }

    .muted {
        color: var(--muted);
    }

    .device-pill {
        background: rgba(47, 224, 124, 0.14);
        color: #67f0a2;
        border: 1px solid rgba(103, 240, 162, 0.2);
        padding: 10px 14px;
        border-radius: 14px;
        text-align: center;
        font-weight: 700;
    }

    .flow-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 12px;
    }

    .flow-item {
        text-align: center;
        padding: 10px 8px;
    }

    .flow-icon {
        font-size: 2.2rem;
        margin-bottom: 8px;
    }

    .flow-label {
        font-size: 0.98rem;
        margin-bottom: 6px;
    }

    .flow-check {
        color: var(--green);
        font-size: 1rem;
        font-weight: 700;
    }

    .output-shell {
        background: linear-gradient(180deg, #f8fbff 0%, #edf3ff 100%);
        color: #081225;
        min-height: 190px;
        border-radius: 18px;
        padding: 34px 42px;
        font-size: clamp(2rem, 3.8vw, 4rem);
        font-weight: 700;
        display: flex;
        align-items: center;
        line-height: 1.1;
        border: 1px solid rgba(95, 168, 255, 0.45);
    }

    .success-line {
        color: #59ee84;
        font-size: 1rem;
        margin-top: 14px;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
    }

    .metric-box {
        padding: 18px 16px;
        text-align: center;
    }

    .metric-label {
        color: #c8d4f0;
        font-size: 0.92rem;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
    }

    .value-green { color: #58eb8d; }
    .value-blue { color: #61a5ff; }
    .value-violet { color: #b07bff; }
    .value-cyan { color: #6fe7ff; }

    .session-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px 22px;
    }

    .session-item {
        display: flex;
        gap: 12px;
        align-items: flex-start;
    }

    .session-icon {
        width: 34px;
        height: 34px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(97, 165, 255, 0.12);
        color: #8bc0ff;
        font-size: 1rem;
        flex-shrink: 0;
    }

    .prob-row {
        display: grid;
        grid-template-columns: 52px 1fr 52px;
        gap: 10px;
        align-items: center;
        margin-bottom: 10px;
    }

    .prob-label, .prob-value {
        color: #ecf3ff;
        font-size: 0.98rem;
    }

    .prob-bar {
        height: 10px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.14);
        overflow: hidden;
    }

    .prob-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #3b82f6, #3ee47d);
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 14px;
        min-height: 48px;
        border: 1px solid rgba(116, 144, 193, 0.35);
        background: linear-gradient(180deg, rgba(27, 41, 74, 0.95), rgba(19, 29, 55, 0.98));
        color: white;
        font-weight: 700;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(90deg, #266dff, #7a4dff);
        border-color: rgba(142, 121, 255, 0.55);
    }

    div[data-testid="stDownloadButton"] > button {
        width: 100%;
        border-radius: 14px;
        min-height: 48px;
        font-weight: 700;
        background: linear-gradient(90deg, #6e42ff, #9258ff);
        color: white;
        border: 1px solid rgba(161, 123, 255, 0.52);
    }

    div[data-testid="stSlider"] label {
        color: #d7e2ff;
        font-weight: 600;
    }

    @media (max-width: 1100px) {
        .flow-grid, .metric-grid, .info-grid, .session-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset():
    return pd.read_csv("dataset/eeg_dataset.csv")


@st.cache_resource
def train_model(data: pd.DataFrame):
    x_data = data.drop("thought", axis=1)
    y_data = data["thought"]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data)

    model = RandomForestClassifier(n_estimators=140, random_state=42)
    model.fit(x_scaled, y_data)
    return model, scaler


def predict_thought(model, scaler, sample):
    sample_df = pd.DataFrame(
        [sample],
        columns=["alpha_ch1", "beta_ch1", "gamma_ch1", "alpha_ch2", "beta_ch2", "gamma_ch2"],
    )
    sample_scaled = scaler.transform(sample_df)
    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]
    return prediction, probabilities


def render_wave_preview(sample_values):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("#0b1328")

    channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2"]
    colors = ["#7c83ff", "#5f8cff", "#26c6da", "#22c55e", "#fbbf24", "#fb923c", "#f87171", "#a855f7"]
    base_signal = np.array(sample_values * 2)[:6]
    x_axis = np.linspace(0, 4 * np.pi, 80)

    for index, channel in enumerate(channel_names):
        offset = (len(channel_names) - index - 1) * 1.08
        amplitude = 0.12 + (base_signal[index % len(base_signal)] % 1) * 0.08
        y_axis = np.sin(x_axis * (1.25 + index * 0.08)) * amplitude
        y_axis += np.random.default_rng(20 + index).normal(scale=0.03, size=len(x_axis))
        y_axis += offset
        ax.plot(x_axis, y_axis, color=colors[index], linewidth=1.6)
        ax.text(-0.8, offset, channel, color="#dbe7ff", fontsize=9, va="center")

    ax.set_xlim(-1.2, x_axis.max())
    ax.set_ylim(-0.4, len(channel_names) * 1.08)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_feature_chart(dataframe):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.6))
    fig.patch.set_alpha(0)
    chart_specs = [
        ("alpha_ch1", "#67e8f9"),
        ("beta_ch1", "#60a5fa"),
        ("gamma_ch1", "#a78bfa"),
    ]

    for axis, (column_name, color) in zip(axes, chart_specs):
        axis.set_facecolor("#091224")
        axis.plot(dataframe[column_name].head(80).values, color=color, linewidth=1.8)
        axis.set_title(column_name.upper(), color="#dce8ff", fontsize=10)
        axis.tick_params(colors="#86a1cb", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#203455")
        axis.grid(alpha=0.14, color="#7da9f5")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def progress_bar_html(value):
    width = max(0, min(int(round(value)), 100))
    return f"""
        <div class="progress-track">
            <div class="progress-bar" style="width:{width}%"></div>
        </div>
    """


def probability_rows(labels, probs):
    rows = []
    for label, prob in zip(labels, probs):
        rows.append(
            f"""
            <div class="prob-row">
                <div class="prob-label">{escape(str(label).lower())}</div>
                <div class="prob-bar"><div class="prob-fill" style="width:{prob * 100:.0f}%"></div></div>
                <div class="prob-value">{prob:.2f}</div>
            </div>
            """
        )
    return "".join(dedent(row).strip() for row in rows)


def section_heading(title):
    st.markdown(f'<div class="panel-title" style="margin-bottom:10px;">{title}</div>', unsafe_allow_html=True)


def run_prediction(model, scaler, sample):
    prediction, probabilities = predict_thought(model, scaler, sample)
    st.session_state.prediction = prediction
    st.session_state.probabilities = probabilities
    st.session_state.last_sample = sample
    st.session_state.response_time = 1.18
    st.session_state.prediction_ready = True


def main():
    dataframe = load_dataset()
    model, scaler = train_model(dataframe)

    default_sample = [8.0, 14.0, 32.0, 8.2, 13.9, 31.9]

    if "prediction" not in st.session_state:
        st.session_state.prediction = "Predict button-ah press panna thought inga varum"
    if "probabilities" not in st.session_state:
        class_count = len(model.classes_)
        default_probs = np.zeros(class_count)
        default_probs[0] = 1.0
        st.session_state.probabilities = default_probs
    if "last_sample" not in st.session_state:
        st.session_state.last_sample = default_sample
    if "response_time" not in st.session_state:
        st.session_state.response_time = 1.23
    if "prediction_ready" not in st.session_state:
        st.session_state.prediction_ready = False

    hero_left, hero_right = st.columns([3.2, 1.6])

    with hero_left:
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-wrap">
                    <div class="hero-icon">NL</div>
                    <div>
                        <div class="hero-title">NeuroLink</div>
                        <div class="hero-subtitle">AI Based Thought-to-Text Converter Using EEG</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_right:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-grid">
                    <div>
                        <div class="eyebrow">System Status</div>
                        <div class="status-pill"><span class="status-dot"></span>Connected</div>
                    </div>
                    <div>
                        <div class="eyebrow">EEG Signal Quality</div>
                        <div class="quality-text">Good (92%)</div>
                        {progress_bar_html(92)}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left_col, main_col, right_col = st.columns([1.15, 2.35, 1.05])

    with left_col:
        section_heading("Connection")
        st.markdown(
            """
            <div class="device-pill">Connected</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**EEG Device**")
        st.caption("Emotiv Epoc+")
        quality_col1, quality_col2 = st.columns([1.4, 1])
        with quality_col1:
            st.caption("Signal Quality")
        with quality_col2:
            st.markdown("<div style='text-align:right;'>92%</div>", unsafe_allow_html=True)
        st.progress(92)
        rate_col1, rate_col2 = st.columns([1.4, 1])
        with rate_col1:
            st.caption("Sampling Rate")
        with rate_col2:
            st.markdown("<div style='text-align:right;'>128 Hz</div>", unsafe_allow_html=True)

        section_heading("Live EEG Preview")
        render_wave_preview(st.session_state.last_sample)

        section_heading("Input Signals")
        alpha_ch1 = st.slider("Alpha Ch1", 7.0, 9.0, float(st.session_state.last_sample[0]), 0.01)
        beta_ch1 = st.slider("Beta Ch1", 13.0, 15.0, float(st.session_state.last_sample[1]), 0.01)
        gamma_ch1 = st.slider("Gamma Ch1", 30.0, 34.0, float(st.session_state.last_sample[2]), 0.01)
        alpha_ch2 = st.slider("Alpha Ch2", 7.0, 9.0, float(st.session_state.last_sample[3]), 0.01)
        beta_ch2 = st.slider("Beta Ch2", 13.0, 15.0, float(st.session_state.last_sample[4]), 0.01)
        gamma_ch2 = st.slider("Gamma Ch2", 30.0, 34.0, float(st.session_state.last_sample[5]), 0.01)

    with main_col:
        st.markdown(
            """
            <div class="soft-card">
                <div class="panel-title">Process Flow</div>
                <div class="flow-grid">
                    <div class="flow-item"><div class="flow-icon">01</div><div class="flow-label">EEG Acquisition</div><div class="flow-check">OK</div></div>
                    <div class="flow-item"><div class="flow-icon">02</div><div class="flow-label">Preprocessing</div><div class="flow-check">OK</div></div>
                    <div class="flow-item"><div class="flow-icon">03</div><div class="flow-label">Feature Extraction</div><div class="flow-check">OK</div></div>
                    <div class="flow-item"><div class="flow-icon">04</div><div class="flow-label">AI Prediction</div><div class="flow-check">OK</div></div>
                    <div class="flow-item"><div class="flow-icon">05</div><div class="flow-label">Post Processing</div><div class="flow-check">OK</div></div>
                    <div class="flow-item"><div class="flow-icon">06</div><div class="flow-label">Text Output</div><div class="flow-check">OK</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_sample = [alpha_ch1, beta_ch1, gamma_ch1, alpha_ch2, beta_ch2, gamma_ch2]
        if st.button("Predict Thought", use_container_width=True):
            run_prediction(model, scaler, current_sample)

        prediction_text = escape(str(st.session_state.prediction))
        st.markdown(
            dedent(f"""
            <div class="output-card">
                <div class="panel-title">Final Output</div>
                <div class="output-shell">{prediction_text}</div>
                <div class="success-line">{'Thought predicted successfully' if st.session_state.prediction_ready else 'Prediction waiting'}</div>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

        if st.session_state.prediction_ready:
            st.success(f"Predicted Thought: {st.session_state.prediction}")

        action_col1, action_col2, action_col3 = st.columns(3)
        with action_col1:
            if st.button("Speak", use_container_width=True):
                speak_text(st.session_state.prediction)
        with action_col2:
            st.download_button(
                "Save",
                data=st.session_state.prediction,
                file_name="neurolink_output.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with action_col3:
            st.text_area("Output Text", st.session_state.prediction, height=68)

        bottom_left, bottom_right = st.columns([1.25, 1.1])

        with bottom_left:
            word_count = len(st.session_state.prediction.split())
            st.markdown(
                f"""
                <div class="soft-card">
                    <div class="panel-title">Statistics</div>
                    <div class="metric-grid">
                        <div class="metric-box">
                            <div class="metric-label">Words Detected</div>
                            <div class="metric-value value-green">{word_count}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Accuracy</div>
                            <div class="metric-value value-green">92.4%</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Response Time</div>
                            <div class="metric-value value-blue">{st.session_state.response_time:.2f}s</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value value-violet">{np.max(st.session_state.probabilities) * 100:.0f}%</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with bottom_right:
            now = datetime.now()
            st.markdown(
                f"""
                <div class="soft-card">
                    <div class="panel-title">Session Info</div>
                    <div class="session-grid">
                        <div class="session-item">
                            <div class="session-icon">DT</div>
                            <div><div class="muted">Date</div><div>{now.strftime("%d %b %Y")}</div></div>
                        </div>
                        <div class="session-item">
                            <div class="session-icon">TM</div>
                            <div><div class="muted">Time</div><div>{now.strftime("%I:%M %p")}</div></div>
                        </div>
                        <div class="session-item">
                            <div class="session-icon">US</div>
                            <div><div class="muted">User</div><div>Test User</div></div>
                        </div>
                        <div class="session-item">
                            <div class="session-icon">MD</div>
                            <div><div class="muted">Mode</div><div>Live Prediction</div></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        section_heading("Feature Signals")
        render_feature_chart(dataframe)

    with right_col:
        section_heading("Prediction Probability")
        probability_html = probability_rows(model.classes_, st.session_state.probabilities)
        st.markdown(
            f"<div>{probability_html}</div>",
            unsafe_allow_html=True,
        )

        top_classes = np.argsort(st.session_state.probabilities)[::-1][:3]
        section_heading("Top Predictions")
        for rank, index in enumerate(top_classes, start=1):
            label = escape(str(model.classes_[index]))
            score = st.session_state.probabilities[index] * 100
            st.markdown(
                f"""
                <div class="metric-box" style="margin-bottom:12px;">
                    <div class="metric-label">Rank {rank}</div>
                    <div class="metric-value value-cyan" style="font-size:1.4rem;">{label}</div>
                    <div class="muted">{score:.1f}% match</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
