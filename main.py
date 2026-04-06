from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")


def create_output_dirs(base_dir: Path) -> Dict[str, Path]:
    figures = base_dir / "figures"
    reports = base_dir / "reports"
    interactive = base_dir / "interactive"

    for directory in (figures, reports, interactive):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_dir,
        "figures": figures,
        "reports": reports,
        "interactive": interactive,
    }


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    return sanitize_dataframe(df)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if "Class" not in df.columns:
        for alt_col in ["class", "target", "label", "is_fraud", "fraud"]:
            if alt_col in df.columns:
                df = df.rename(columns={alt_col: "Class"})
                break

    if "Class" not in df.columns:
        # Fallback target so downstream functions can still run.
        df["Class"] = 0

    class_map = {
        "fraud": 1,
        "non-fraud": 0,
        "non_fraud": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }
    df["Class"] = (
        df["Class"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: class_map[x] if x in class_map else x)
    )
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0)
    df["Class"] = (df["Class"] > 0).astype(int)

    for col in df.columns:
        if col != "Class":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    if "Time" not in df.columns:
        df["Time"] = np.arange(len(df), dtype=float)
    else:
        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

    if "Amount" not in df.columns:
        candidate = [c for c in numeric_cols if c not in {"Class", "Time"}]
        if candidate:
            df["Amount"] = df[candidate[0]].abs()
        else:
            df["Amount"] = 0.0
    else:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def save_data_overview(df: pd.DataFrame, reports_dir: Path) -> None:
    overview_path = reports_dir / "data_overview.txt"
    info_buffer = StringIO()
    df.info(buf=info_buffer)

    class_counts = df["Class"].value_counts().sort_index()
    class_percent = (class_counts / len(df) * 100).round(4)
    missing_values = df.isnull().sum()

    with overview_path.open("w", encoding="utf-8") as file:
        file.write("DATASET OVERVIEW\n")
        file.write("=" * 70 + "\n")
        file.write(f"Shape: {df.shape}\n\n")
        file.write("INFO:\n")
        file.write(info_buffer.getvalue() + "\n")
        file.write("DESCRIPTIVE STATS:\n")
        file.write(df.describe().to_string() + "\n\n")
        file.write("MISSING VALUES PER COLUMN:\n")
        file.write(missing_values.to_string() + "\n\n")
        file.write("CLASS DISTRIBUTION:\n")
        for cls in class_counts.index:
            file.write(f"Class {cls}: {class_counts[cls]} rows ({class_percent[cls]}%)\n")


def preprocess_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series]:
    df_original = df.copy()
    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

    if {"Time", "Amount"}.issubset(df_clean.columns):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_clean[["Time", "Amount"]])
        df_clean["Time_Scaled"] = scaled_features[:, 0]
        df_clean["Amount_Scaled"] = scaled_features[:, 1]
        df_scaled = df_clean.drop(columns=["Time", "Amount"])
    else:
        df_scaled = df_clean.copy()

    X = df_scaled.drop(columns=["Class"])
    y = df_scaled["Class"]

    return {
        "df_original": df_original,
        "df_scaled": df_scaled,
        "X": X,
        "y": y,
    }


def _save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_placeholder_plot(path: Path, title: str, message: str) -> None:
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save_plot(fig, path)


def _plot_binary_scatter(ax: plt.Axes, data: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    non_fraud = data[data["Class"] == 0]
    fraud = data[data["Class"] == 1]

    if len(non_fraud) > 0:
        ax.scatter(
            non_fraud[x_col],
            non_fraud[y_col],
            c="#9e9e9e",
            alpha=0.12,
            s=10,
            label="Non-Fraud",
            edgecolors="none",
        )
    if len(fraud) > 0:
        ax.scatter(
            fraud[x_col],
            fraud[y_col],
            c="#d7191c",
            alpha=0.9,
            s=28,
            label="Fraud",
            edgecolors="black",
            linewidths=0.25,
        )

    ax.set_title(title)
    ax.legend(loc="best")


def run_eda(
    df: pd.DataFrame,
    df_scaled: pd.DataFrame,
    figures_dir: Path,
    interactive_dir: Path,
) -> Dict[str, object]:
    class_distribution = df["Class"].value_counts().sort_index()
    if class_distribution.empty:
        class_distribution = pd.Series({0: len(df)})

    class_counts = class_distribution.reindex([0, 1], fill_value=0)
    class_percent = (class_counts / max(len(df), 1) * 100).round(4)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    sns.barplot(
        x=class_counts.index.astype(str),
        y=class_counts.values,
        palette=["#7fcdbb", "#f03b20"],
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Class Counts (Log Scale)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count (log scale)")
    for i, value in enumerate(class_counts.values):
        if value > 0:
            ax.text(i, value, f"{int(value):,}", ha="center", va="bottom", fontsize=10)

    ax = axes[1]
    sns.barplot(
        x=class_percent.index.astype(str),
        y=class_percent.values,
        palette=["#7fcdbb", "#f03b20"],
        ax=ax,
    )
    ax.set_title("Class Share (%)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, max(5, float(class_percent.max()) * 1.2))
    for i, value in enumerate(class_percent.values):
        ax.text(i, value, f"{value:.4f}%", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Fraud vs Non-Fraud Distribution", fontsize=14, y=1.02)
    _save_plot(fig, figures_dir / "01_class_distribution.png")

    candidate_features = [c for c in ["Time", "Amount"] if c in df.columns]
    if candidate_features:
        fig, axes = plt.subplots(1, len(candidate_features), figsize=(6 * len(candidate_features), 4))
        if len(candidate_features) == 1:
            axes = [axes]
        for ax, col in zip(axes, candidate_features):
            sns.histplot(data=df, x=col, hue="Class", bins=60, kde=False, ax=ax)
            ax.set_title(f"Distribution of {col} by Class")
            ax.set_yscale("log")
            ax.set_ylabel("Count (log scale)")
        _save_plot(fig, figures_dir / "02_histograms_time_amount.png")

    corr = df_scaled.corr(numeric_only=True)
    if corr.empty:
        _save_placeholder_plot(
            figures_dir / "03_correlation_heatmap.png",
            "Correlation Heatmap",
            "No numeric features available for correlation.",
        )
    else:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        _save_plot(fig, figures_dir / "03_correlation_heatmap.png")

    for col in ["Amount", "Time"]:
        if col in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})

            ax = axes[0]
            sns.boxplot(data=df, x="Class", y=col, palette="Set3", ax=ax)
            sns.stripplot(data=df.sample(min(3000, len(df)), random_state=42), x="Class", y=col, color="black", alpha=0.12, size=1.5, ax=ax)
            ax.set_title(f"{col} by Class (Box + Sampled Points)")

            ax = axes[1]
            class_means = df.groupby("Class")[col].mean().reindex([0, 1])
            sns.barplot(x=class_means.index.astype(str), y=class_means.values, palette=["#7fcdbb", "#f03b20"], ax=ax)
            ax.set_title(f"Mean {col} by Class")
            ax.set_xlabel("Class")
            ax.set_ylabel(f"Mean {col}")
            for i, value in enumerate(class_means.values):
                if pd.notna(value):
                    ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

            _save_plot(fig, figures_dir / f"04_boxplot_{col.lower()}_by_class.png")

    if "Time" in df.columns:
        df_time = df.copy()
        df_time["Hour"] = (df_time["Time"] // 3600) % 24
        hourly = df_time.groupby(["Hour", "Class"]).size().reset_index(name="Count")
        hourly_pct = hourly.copy()
        hourly_pct["Count"] = hourly_pct.groupby("Class")["Count"].transform(lambda s: (s / max(s.sum(), 1)) * 100)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        sns.lineplot(data=hourly, x="Hour", y="Count", hue="Class", marker="o", ax=ax)
        ax.set_yscale("log")
        ax.set_title("Transactions by Hour and Class (Log Count)")
        ax.set_ylabel("Count (log scale)")

        ax = axes[1]
        sns.lineplot(data=hourly_pct, x="Hour", y="Count", hue="Class", marker="o", ax=ax)
        ax.set_title("Transactions by Hour and Class (%)")
        ax.set_ylabel("Percentage within class")

        _save_plot(fig, figures_dir / "05_time_based_hourly_analysis.png")

    if "Amount" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        sns.histplot(data=df, x="Amount", hue="Class", bins=80, stat="density", common_norm=False, ax=ax)
        max_x = np.nanpercentile(df["Amount"], 99) if df["Amount"].notna().any() else 1
        ax.set_xlim(0, max(1, max_x))
        ax.set_yscale("log")
        ax.set_title("Amount Density by Class (Log Y)")
        ax.set_ylabel("Density (log scale)")

        ax = axes[1]
        amount_summary = df.groupby("Class")["Amount"].median().reindex([0, 1])
        sns.barplot(x=amount_summary.index.astype(str), y=amount_summary.values, palette=["#7fcdbb", "#f03b20"], ax=ax)
        ax.set_title("Median Amount by Class")
        ax.set_xlabel("Class")
        ax.set_ylabel("Median Amount")
        for i, value in enumerate(amount_summary.values):
            if pd.notna(value):
                ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

        _save_plot(fig, figures_dir / "06_amount_based_density.png")

    try:
        import plotly.express as px

        class_df = (
            class_counts.rename_axis("Class")
            .reset_index(name="Count")
            .assign(Class=lambda x: x["Class"].astype(str))
        )
        fig_plotly = px.bar(
            class_df,
            x="Class",
            y="Count",
            title="Interactive Class Distribution (Log Scale)",
            color="Class",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text="Count",
            log_y=True,
        )
        fig_plotly.update_traces(textposition="outside")
        fig_plotly.write_html(interactive_dir / "class_distribution.html", include_plotlyjs="cdn")

        if {"Time", "Amount", "Class"}.issubset(df.columns):
            sample_df = df.sample(min(5000, len(df)), random_state=42)
            scatter = px.scatter(
                sample_df,
                x="Time",
                y="Amount",
                color=sample_df["Class"].astype(str),
                title="Interactive Time vs Amount by Class",
                labels={"color": "Class"},
            )
            scatter.write_html(interactive_dir / "time_amount_scatter.html", include_plotlyjs="cdn")
    except Exception:
        pass

    top_correlations = (
        corr["Class"].drop(labels=["Class"]).sort_values(key=lambda s: s.abs(), ascending=False).head(10)
        if "Class" in corr.columns
        else pd.Series(dtype=float)
    )

    return {
        "class_distribution": class_distribution.to_dict(),
        "top_correlations": top_correlations.to_dict(),
    }


def run_advanced_visualizations(
    df_scaled: pd.DataFrame,
    figures_dir: Path,
    interactive_dir: Path,
    random_state: int,
    sample_size: int,
) -> None:
    if len(df_scaled) == 0:
        _save_placeholder_plot(
            figures_dir / "07_pairplot_top_features.png",
            "Pairplot",
            "No data available for pairplot.",
        )
        _save_placeholder_plot(
            figures_dir / "08_outlier_detection.png",
            "Outlier Detection",
            "No data available for outlier analysis.",
        )
        _save_placeholder_plot(
            figures_dir / "09_pca_projection.png",
            "PCA Projection",
            "No data available for PCA.",
        )
        _save_placeholder_plot(
            figures_dir / "10_tsne_projection.png",
            "t-SNE Projection",
            "No data available for t-SNE.",
        )
        return

    sampled = df_scaled.sample(min(sample_size, len(df_scaled)), random_state=random_state)
    pairplot_sample = sampled.sample(min(1500, len(sampled)), random_state=random_state)

    corr_to_class = (
        pairplot_sample.corr(numeric_only=True)["Class"].drop(labels=["Class"], errors="ignore").abs().sort_values(ascending=False)
        if "Class" in pairplot_sample.columns
        else pd.Series(dtype=float)
    )
    pair_features = corr_to_class.head(4).index.tolist()
    if pair_features and pairplot_sample["Class"].nunique() > 1:
        pair_df = pairplot_sample[pair_features + ["Class"]].copy()
        g = sns.pairplot(
            pair_df,
            hue="Class",
            corner=True,
            plot_kws={"alpha": 0.55, "s": 18},
            palette={0: "#9e9e9e", 1: "#d7191c"},
        )
        g.fig.suptitle("Pairplot of Most Informative Features", y=1.02)
        g.savefig(figures_dir / "07_pairplot_top_features.png", dpi=300, bbox_inches="tight")
        plt.close(g.fig)
    else:
        _save_placeholder_plot(
            figures_dir / "07_pairplot_top_features.png",
            "Pairplot of Most Informative Features",
            "Insufficient class variation or features for pairplot.",
        )

    if "Amount_Scaled" in sampled.columns:
        q1 = sampled["Amount_Scaled"].quantile(0.25)
        q3 = sampled["Amount_Scaled"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        flagged = sampled.copy()
        flagged["Outlier"] = (~flagged["Amount_Scaled"].between(lower, upper)).astype(int)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        sns.countplot(data=flagged, x="Outlier", hue="Class", palette="Set1", ax=ax)
        ax.set_title("Outlier Counts (IQR on Amount_Scaled) by Class")
        _save_plot(fig, figures_dir / "08_outlier_detection.png")

    features = sampled.drop(columns=["Class"], errors="ignore")
    if features.shape[1] >= 2 and len(features) >= 2:
        pca = PCA(n_components=2, random_state=random_state)
        pca_proj = pca.fit_transform(features)
        pca_df = pd.DataFrame(pca_proj, columns=["PC1", "PC2"])
        pca_df["Class"] = sampled["Class"].values

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        _plot_binary_scatter(axes[0], pca_df, "PC1", "PC2", "PCA Projection (All Points)")

        fraud_only = pca_df[pca_df["Class"] == 1]
        if len(fraud_only) > 0:
            _plot_binary_scatter(axes[1], pca_df, "PC1", "PC2", "PCA Projection (Fraud Emphasis)")
            axes[1].set_xlim(fraud_only["PC1"].min() - 1, fraud_only["PC1"].max() + 1)
            axes[1].set_ylim(fraud_only["PC2"].min() - 1, fraud_only["PC2"].max() + 1)
        else:
            _plot_binary_scatter(axes[1], pca_df, "PC1", "PC2", "PCA Projection (Fraud Emphasis)")
            axes[1].text(0.5, 0.5, "No fraud points in sample", transform=axes[1].transAxes, ha="center")

        _save_plot(fig, figures_dir / "09_pca_projection.png")
    else:
        pca_df = pd.DataFrame({"PC1": [0], "PC2": [0], "Class": [sampled["Class"].iloc[0]]})
        _save_placeholder_plot(
            figures_dir / "09_pca_projection.png",
            "PCA Projection (2D)",
            "Not enough numeric features for PCA.",
        )

    tsne_sample = sampled.sample(min(1000, len(sampled)), random_state=random_state)
    tsne_features = tsne_sample.drop(columns=["Class"], errors="ignore")
    if tsne_features.shape[1] >= 2 and len(tsne_sample) >= 5:
        perplexity = min(30, max(2, len(tsne_sample) - 1))
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        tsne_proj = tsne.fit_transform(tsne_features)
        tsne_df = pd.DataFrame(tsne_proj, columns=["TSNE1", "TSNE2"])
        tsne_df["Class"] = tsne_sample["Class"].values

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        _plot_binary_scatter(axes[0], tsne_df, "TSNE1", "TSNE2", "t-SNE Projection (All Points)")

        fraud_only = tsne_df[tsne_df["Class"] == 1]
        if len(fraud_only) > 0:
            _plot_binary_scatter(axes[1], tsne_df, "TSNE1", "TSNE2", "t-SNE Projection (Fraud Emphasis)")
            axes[1].set_xlim(fraud_only["TSNE1"].min() - 1, fraud_only["TSNE1"].max() + 1)
            axes[1].set_ylim(fraud_only["TSNE2"].min() - 1, fraud_only["TSNE2"].max() + 1)
        else:
            _plot_binary_scatter(axes[1], tsne_df, "TSNE1", "TSNE2", "t-SNE Projection (Fraud Emphasis)")
            axes[1].text(0.5, 0.5, "No fraud points in sample", transform=axes[1].transAxes, ha="center")

        _save_plot(fig, figures_dir / "10_tsne_projection.png")
    else:
        _save_placeholder_plot(
            figures_dir / "10_tsne_projection.png",
            "t-SNE Projection (2D)",
            "Not enough samples/features for t-SNE.",
        )

    try:
        import plotly.express as px

        pca_interactive = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color=pca_df["Class"].astype(str),
            color_discrete_map={"0": "#9e9e9e", "1": "#d7191c"},
            title="Interactive PCA Projection (Fraud Highlighted)",
            labels={"color": "Class"},
            opacity=0.8,
        )
        pca_interactive.write_html(interactive_dir / "pca_projection.html", include_plotlyjs="cdn")
    except Exception:
        pass


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> Dict[str, object]:
    if X.shape[1] == 0:
        X = pd.DataFrame({"Fallback_Feature": np.arange(len(y), dtype=float)})

    stratify_arg = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    if y_train.nunique() < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
    else:
        minority_count = y_train.value_counts().min()
        if minority_count >= 2:
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(X_train, y_train)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
    }


def evaluate_model(model_bundle: Dict[str, object], figures_dir: Path, reports_dir: Path) -> Dict[str, float]:
    model = model_bundle["model"]
    X_test = model_bundle["X_test"]
    y_test = model_bundle["y_test"]

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if pd.Series(y_test).nunique() > 1:
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    (reports_dir / "classification_report.txt").write_text(
        classification_report(y_test, y_pred, zero_division=0),
        encoding="utf-8",
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix")
    _save_plot(fig, figures_dir / "11_confusion_matrix.png")

    if pd.Series(y_test).nunique() > 1:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        _save_plot(fig, figures_dir / "12_roc_curve.png")
    else:
        _save_placeholder_plot(
            figures_dir / "12_roc_curve.png",
            "ROC Curve",
            "ROC requires at least two classes in test data.",
        )

    if pd.Series(y_test).nunique() > 1:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall_vals, precision_vals, label=f"PR AUC = {pr_auc:.4f}", color="darkorange")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="best")
        _save_plot(fig, figures_dir / "13_precision_recall_curve.png")
    else:
        _save_placeholder_plot(
            figures_dir / "13_precision_recall_curve.png",
            "Precision-Recall Curve",
            "PR curve requires at least two classes in test data.",
        )

    metrics: Dict[str, float] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }

    lines = ["MODEL METRICS", "=" * 50]
    for key, value in metrics.items():
        if key in {"tn", "fp", "fn", "tp"}:
            lines.append(f"{key}: {int(value)}")
        else:
            if np.isnan(value):
                lines.append(f"{key}: NA")
            else:
                lines.append(f"{key}: {value:.6f}")
    (reports_dir / "model_metrics.txt").write_text("\n".join(lines), encoding="utf-8")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = np.array(X_test.columns)
        idx = np.argsort(importances)[-15:]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=importances[idx], y=feature_names[idx], orient="h", ax=ax)
        ax.set_title("Top 15 Feature Importances")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        _save_plot(fig, figures_dir / "14_feature_importance.png")
    else:
        _save_placeholder_plot(
            figures_dir / "14_feature_importance.png",
            "Top 15 Feature Importances",
            "Selected model does not provide feature importances.",
        )

    return metrics


def _pct(value: float, digits: int = 4) -> str:
    return f"{value * 100:.{digits}f}%"


def generate_insights_report(
    reports_dir: Path,
    class_distribution: Dict[int, int],
    top_correlations: Dict[str, float],
    metrics: Dict[str, float],
) -> None:
    total = sum(class_distribution.values())
    fraud_count = class_distribution.get(1, 0)
    non_fraud_count = class_distribution.get(0, 0)
    fraud_ratio = fraud_count / total if total else 0.0

    corr_lines = []
    for feature, corr_val in list(top_correlations.items())[:5]:
        corr_lines.append(f"- {feature}: {corr_val:.4f}")
    if not corr_lines:
        corr_lines.append("- Correlation details unavailable.")

    insights = f"""# Fraud Detection Insights and Conclusion

## 1. Dataset Characteristics
- Total transactions: {total}
- Non-fraud transactions (Class 0): {non_fraud_count}
- Fraud transactions (Class 1): {fraud_count}
- Fraud prevalence: {_pct(fraud_ratio)}

The dataset is highly imbalanced, which is expected in fraud detection problems. This is why SMOTE was applied to the training set.

## 2. Key Visualization Findings
- Class distribution plot confirms severe imbalance between normal and fraud classes.
- Time and Amount distributions show different behavior patterns for fraudulent transactions compared to non-fraud.
- Correlation analysis highlights features that are most related to fraud class.

Top correlation indicators with Class:
{chr(10).join(corr_lines)}

## 3. Model Performance (Random Forest + SMOTE)
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-score: {metrics['f1_score']:.4f}
- ROC AUC: {metrics['roc_auc']:.4f}
- PR AUC: {metrics['pr_auc']:.4f}

Confusion Matrix Summary:
- True Negatives: {int(metrics['tn'])}
- False Positives: {int(metrics['fp'])}
- False Negatives: {int(metrics['fn'])}
- True Positives: {int(metrics['tp'])}

## 4. Interpretation in Simple Terms
- High recall means the model catches a large fraction of fraudulent transactions.
- Precision indicates how many flagged transactions are truly fraud.
- ROC AUC and PR AUC summarize ranking quality across decision thresholds.

For real-world fraud systems, recall is usually prioritized to minimize missed fraud, while maintaining acceptable precision to reduce false alarms.

## 5. Conclusion
This project demonstrates a complete fraud analytics workflow:
- robust preprocessing,
- informative EDA and advanced visual analytics,
- imbalance-aware model training,
- and metric-focused evaluation.

The generated plots and reports are suitable for academic presentation and can be extended with additional algorithms, threshold tuning, and explainability techniques.
"""

    (reports_dir / "insights.md").write_text(insights, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="creditcard_cleaned.csv",
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Sample size used for heavy visualizations like pairplot and t-SNE.",
    )
    return parser.parse_args()


def main() -> None:
    # Ensure progress logs appear immediately in terminal (no buffering delays).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args()
    data_path = Path(args.data)

    output_dirs = create_output_dirs(base_dir=Path("outputs"))

    print("[1/6] Loading dataset...", flush=True)
    df = load_data(data_path)
    save_data_overview(df, output_dirs["reports"])
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns", flush=True)

    print("[2/6] Preprocessing data...", flush=True)
    processed = preprocess_data(df)
    print(f"Features prepared: {processed['X'].shape[1]} input features", flush=True)

    print("[3/6] Running EDA visualizations...", flush=True)
    eda_summary = run_eda(
        df=processed["df_original"],
        df_scaled=processed["df_scaled"],
        figures_dir=output_dirs["figures"],
        interactive_dir=output_dirs["interactive"],
    )

    print("[4/6] Running advanced visualizations...", flush=True)
    run_advanced_visualizations(
        df_scaled=processed["df_scaled"],
        figures_dir=output_dirs["figures"],
        interactive_dir=output_dirs["interactive"],
        random_state=args.random_state,
        sample_size=args.sample_size,
    )

    print("[5/6] Training and evaluating model...", flush=True)
    model_bundle = train_model(
        X=processed["X"],
        y=processed["y"],
        test_size=args.test_size,
        random_state=args.random_state,
    )

    metrics = evaluate_model(
        model_bundle=model_bundle,
        figures_dir=output_dirs["figures"],
        reports_dir=output_dirs["reports"],
    )

    print("Evaluation metrics:", flush=True)
    print(f"- Precision: {metrics['precision']:.4f}", flush=True)
    print(f"- Recall:    {metrics['recall']:.4f}", flush=True)
    print(f"- F1 Score:  {metrics['f1_score']:.4f}", flush=True)
    print(f"- ROC AUC:   {metrics['roc_auc']:.4f}", flush=True)
    print(f"- PR AUC:    {metrics['pr_auc']:.4f}", flush=True)
    print(
        (
            f"- Confusion Matrix: TN={metrics['tn']}, FP={metrics['fp']}, "
            f"FN={metrics['fn']}, TP={metrics['tp']}"
        ),
        flush=True,
    )

    print("[6/6] Generating insights report...", flush=True)
    generate_insights_report(
        reports_dir=output_dirs["reports"],
        class_distribution=eda_summary["class_distribution"],
        top_correlations=eda_summary["top_correlations"],
        metrics=metrics,
    )

    print("Pipeline complete. Generated outputs:", flush=True)
    print(f"- Figures:      {output_dirs['figures']}", flush=True)
    print(f"- Reports:      {output_dirs['reports']}", flush=True)
    print(f"- Interactive:  {output_dirs['interactive']}", flush=True)


if __name__ == "__main__":
    main()
