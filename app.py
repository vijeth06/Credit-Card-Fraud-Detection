from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError

from main import (
    create_output_dirs,
    evaluate_model,
    generate_insights_report,
    preprocess_data,
    run_advanced_visualizations,
    run_eda,
    sanitize_dataframe,
    train_model,
)


st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #102a43 0%, #1f6feb 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(16, 42, 67, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1, .hero p {
        margin: 0;
    }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(31, 111, 235, 0.12);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>💳 Credit Card Fraud Detection Dashboard</h1>
        <p>Interactive exploration, model training, evaluation, and report generation in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def load_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return sanitize_dataframe(df)

    default_path = Path("creditcard.csv")
    if default_path.exists():
        return sanitize_dataframe(pd.read_csv(default_path))

    st.error("No dataset found. Upload a CSV file or keep creditcard.csv in the project folder.")
    st.stop()


def render_image_safe(image_path: Path, caption: str) -> None:
    try:
        with Image.open(image_path) as image_file:
            image_file.load()
            st.image(image_file.copy(), caption=caption, use_container_width=True)
    except (UnidentifiedImageError, OSError) as error:
        st.warning(f"Could not display {caption}: {error}")


with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload credit card dataset", type=["csv"])
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    sample_size = st.slider("Visualization sample size", 1000, 20000, 10000, 1000)
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)


if "results" not in st.session_state:
    st.session_state.results = None


if run_button:
    with st.spinner("Loading dataset and running the full analysis pipeline..."):
        df = load_dataset(uploaded)
        output_dirs = create_output_dirs(Path("outputs"))
        processed = preprocess_data(df)

        eda_summary = run_eda(
            df=processed["df_original"],
            df_scaled=processed["df_scaled"],
            figures_dir=output_dirs["figures"],
            interactive_dir=output_dirs["interactive"],
        )

        run_advanced_visualizations(
            df_scaled=processed["df_scaled"],
            figures_dir=output_dirs["figures"],
            interactive_dir=output_dirs["interactive"],
            random_state=int(random_state),
            sample_size=int(sample_size),
        )

        model_bundle = train_model(
            X=processed["X"],
            y=processed["y"],
            test_size=float(test_size),
            random_state=int(random_state),
        )

        metrics = evaluate_model(
            model_bundle=model_bundle,
            figures_dir=output_dirs["figures"],
            reports_dir=output_dirs["reports"],
        )

        generate_insights_report(
            reports_dir=output_dirs["reports"],
            class_distribution=eda_summary["class_distribution"],
            top_correlations=eda_summary["top_correlations"],
            metrics=metrics,
        )

        st.session_state.results = {
            "df": df,
            "processed": processed,
            "eda_summary": eda_summary,
            "metrics": metrics,
            "output_dirs": output_dirs,
        }


results = st.session_state.results

if results is None:
    st.info("Upload a CSV file or use creditcard.csv, then click **Run Analysis** to generate the full dashboard.")
    st.stop()


df = results["df"]
processed = results["processed"]
eda_summary = results["eda_summary"]
metrics = results["metrics"]
output_dirs = results["output_dirs"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{df.shape[0]:,}")
col2.metric("Columns", f"{df.shape[1]:,}")
col3.metric("Fraud Cases", f"{eda_summary['class_distribution'].get(1, 0):,}")
col4.metric("Fraud Rate", f"{(eda_summary['class_distribution'].get(1, 0) / max(df.shape[0], 1)) * 100:.4f}%")


tab_overview, tab_eda, tab_model, tab_reports = st.tabs(["Overview", "EDA", "Model", "Reports"])

with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Class Distribution")
        st.bar_chart(pd.Series(eda_summary["class_distribution"]))
    with right:
        st.markdown("### Summary")
        st.write(df.describe(include="all").transpose())


with tab_eda:
    st.subheader("Static Figures")
    figure_files = [
        "01_class_distribution.png",
        "02_histograms_time_amount.png",
        "03_correlation_heatmap.png",
        "04_boxplot_amount_by_class.png",
        "04_boxplot_time_by_class.png",
        "05_time_based_hourly_analysis.png",
        "06_amount_based_density.png",
        "07_pairplot_top_features.png",
        "08_outlier_detection.png",
        "09_pca_projection.png",
        "10_tsne_projection.png",
    ]

    for figure_name in figure_files:
        figure_path = output_dirs["figures"] / figure_name
        if figure_path.exists():
            render_image_safe(figure_path, figure_name)

    st.subheader("Interactive Charts")
    interactive_files = [
        "class_distribution.html",
        "time_amount_scatter.html",
        "pca_projection.html",
    ]

    for html_name in interactive_files:
        html_path = output_dirs["interactive"] / html_name
        if html_path.exists():
            st.markdown(f"#### {html_name}")
            components.html(html_path.read_text(encoding="utf-8"), height=600, scrolling=True)


with tab_model:
    st.subheader("Model Metrics")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Precision", f"{metrics['precision']:.4f}")
    metric_cols[1].metric("Recall", f"{metrics['recall']:.4f}")
    metric_cols[2].metric("F1 Score", f"{metrics['f1_score']:.4f}")
    metric_cols[3].metric("ROC AUC", f"{metrics['roc_auc']:.4f}" if pd.notna(metrics['roc_auc']) else "NA")
    metric_cols[4].metric("PR AUC", f"{metrics['pr_auc']:.4f}" if pd.notna(metrics['pr_auc']) else "NA")

    st.markdown("### Confusion Matrix and Curves")
    model_figs = ["11_confusion_matrix.png", "12_roc_curve.png", "13_precision_recall_curve.png", "14_feature_importance.png"]
    for figure_name in model_figs:
        figure_path = output_dirs["figures"] / figure_name
        if figure_path.exists():
            st.image(str(figure_path), caption=figure_name, use_container_width=True)


with tab_reports:
    st.subheader("Generated Reports")
    report_files = [
        "data_overview.txt",
        "classification_report.txt",
        "model_metrics.txt",
        "insights.md",
    ]

    for report_name in report_files:
        report_path = output_dirs["reports"] / report_name
        if report_path.exists():
            st.markdown(f"### {report_name}")
            st.code(report_path.read_text(encoding="utf-8"), language="text")
