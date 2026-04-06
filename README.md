# Credit Card Fraud Detection and Visual Analytics

An end-to-end machine learning and data visualization project for detecting fraudulent credit card transactions. This repository is designed to be clean, presentable, and easy to share on GitHub.

## ✨ Highlights

- Single-file implementation in [main.py](main.py)
- Full workflow: preprocessing, EDA, advanced visualization, modeling, evaluation, and reporting
- Imbalance-aware training using SMOTE on training data only
- Publication-style static figures and interactive HTML charts
- Clean output reports for academic submission and documentation

## 🎯 Why This Project

Fraud detection is a highly imbalanced classification problem. In this setting, accuracy alone is not enough, so the project focuses on recall, PR AUC, ROC AUC, and confusion-matrix interpretation.

## 📦 Dataset

The dataset should contain the following columns:

- Time
- Amount
- anonymized features such as V1 to V28
- Class, where `0 = Non-Fraud` and `1 = Fraud`

Default input file:

- [creditcard.csv](creditcard.csv)

## 🛠️ Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn, plotly
- scikit-learn
- imbalanced-learn

## 🧱 Project Structure

Current repository layout:

```text
DV/
├── main.py
├── creditcard.csv
└── README.md
```

Generated when you run the project:

```text
outputs/
├── figures/
├── reports/
└── interactive/
```

## 🚀 How to Run

Install the dependencies in your current Python environment:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn plotly kaleido
```

Run the full pipeline with the default dataset:

```bash
python main.py
```

Run with explicit arguments:

```bash
python main.py --data creditcard.csv --test-size 0.2 --random-state 42 --sample-size 10000
```

## ⚙️ Command-Line Arguments

- `--data`: path to the CSV dataset
- `--test-size`: test split ratio, default `0.2`
- `--random-state`: seed for reproducibility, default `42`
- `--sample-size`: sample size for heavier visualizations, default `10000`

## 📈 What the Pipeline Produces

### Reports

- outputs/reports/data_overview.txt
- outputs/reports/classification_report.txt
- outputs/reports/model_metrics.txt
- outputs/reports/insights.md

### Static Figures

- outputs/figures/01_class_distribution.png
- outputs/figures/02_histograms_time_amount.png
- outputs/figures/03_correlation_heatmap.png
- outputs/figures/04_boxplot_amount_by_class.png
- outputs/figures/04_boxplot_time_by_class.png
- outputs/figures/05_time_based_hourly_analysis.png
- outputs/figures/06_amount_based_density.png
- outputs/figures/07_pairplot_top_features.png
- outputs/figures/08_outlier_detection.png
- outputs/figures/09_pca_projection.png
- outputs/figures/10_tsne_projection.png
- outputs/figures/11_confusion_matrix.png
- outputs/figures/12_roc_curve.png
- outputs/figures/13_precision_recall_curve.png
- outputs/figures/14_feature_importance.png

### Interactive Charts

- outputs/interactive/class_distribution.html
- outputs/interactive/time_amount_scatter.html
- outputs/interactive/pca_projection.html

## 🧠 Pipeline Summary

1. Data understanding and quality checks
- Loads the CSV file
- Validates the target column
- Summarizes schema, descriptive statistics, and class balance

2. Preprocessing
- Handles missing numeric values with median imputation
- Scales Time and Amount with StandardScaler
- Prepares model features and target labels

3. Exploratory data analysis
- Class distribution visualization
- Time and amount distribution analysis
- Correlation heatmap
- Boxplots and trend plots

4. Advanced visualization
- Pairplot of informative features
- Outlier analysis using IQR
- PCA visualization
- t-SNE visualization
- Interactive Plotly outputs

5. Modeling
- Stratified train/test split
- SMOTE oversampling on training data only
- Random Forest classifier

6. Evaluation and reporting
- Confusion matrix
- Precision, Recall, and F1-score
- ROC curve and ROC AUC
- Precision-Recall curve and PR AUC
- Feature importance chart
- Written insights report

## ✅ Reproducibility Notes

- The default random seed is `42`
- SMOTE is applied only on the training set
- The project is optimized for imbalanced fraud detection datasets
- If runtime is high, reduce `--sample-size`

## 🧹 Git-Friendly Workflow

- Keep [main.py](main.py), [README.md](README.md), and [creditcard.csv](creditcard.csv) in the repository
- Regenerate the `outputs/` folder by rerunning the script when needed
- Avoid committing large generated artifacts unless your submission requires them

## 🛟 Troubleshooting

- If imports fail, reinstall dependencies using the install command above
- If the dataset path is different, pass it through `--data`
- If plots take too long, lower the sample size, for example `3000`

## 🔮 Future Improvements

- Add model comparison across multiple algorithms
- Add threshold tuning for better recall-precision trade-offs
- Add SHAP-based explainability
- Add a simple dashboard for interactive analysis

## 📄 License

Intended for academic and educational use.
