import io
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


DATA_PATH = Path("data/citrus.csv")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def setup_output_dir():
    """Membuat folder outputs jika belum ada."""
    OUTPUT_DIR.mkdir(exist_ok=True)


def save_text(filename, content):
    """Menyimpan teks ke file outputs."""
    file_path = OUTPUT_DIR / filename
    file_path.write_text(content, encoding="utf-8")


def load_data(file_path):
    """Membaca dataset dan membersihkan nama kolom."""
    if not file_path.exists():
        raise FileNotFoundError(
            "Dataset tidak ditemukan. Pastikan file berada di data/citrus.csv"
        )

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    required_columns = {"name", "diameter", "weight", "red", "green", "blue"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Kolom berikut tidak ditemukan: {missing_columns}")

    return df


def business_understanding():
    """Menyimpan penjelasan Business Understanding."""
    text = (
        "BUSINESS UNDERSTANDING\n"
        "======================\n\n"
        "Tujuan project ini adalah membuat model klasifikasi untuk membedakan "
        "buah orange dan grapefruit berdasarkan fitur numerik pada dataset.\n\n"
        "Model yang dibandingkan:\n"
        "1. Decision Tree\n"
        "2. Naive Bayes\n"
        "3. Support Vector Machine\n"
    )

    print(text)
    save_text("business_understanding.txt", text)


def data_understanding(df):
    """Melakukan eksplorasi awal dataset dan menyimpan hasilnya."""
    print("\nDATA UNDERSTANDING")
    print("==================")

    print("\n1. Lima data pertama:")
    print(df.head())

    df.head().to_csv(OUTPUT_DIR / "data_understanding_5_data_pertama.csv", index=False)

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()
    save_text("data_understanding_info_dataset.txt", info_text)

    df.describe().to_csv(OUTPUT_DIR / "data_understanding_statistik_deskriptif.csv")

    df.isnull().sum().to_csv(
        OUTPUT_DIR / "data_understanding_missing_values.csv",
        header=["jumlah_missing"],
    )

    df["name"].value_counts().to_csv(
        OUTPUT_DIR / "data_understanding_distribusi_kelas.csv",
        header=["jumlah"],
    )

    summary = (
        f"Jumlah data: {len(df)}\n"
        f"Jumlah kolom: {len(df.columns)}\n"
        f"Jumlah duplikat: {df.duplicated().sum()}\n"
        f"Nama kolom: {list(df.columns)}\n"
    )

    print("\nRingkasan dataset:")
    print(summary)
    save_text("data_understanding_ringkasan.txt", summary)


def plot_data_visualization(df):
    """Membuat visualisasi awal dataset."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="name")
    plt.title("Distribusi Kelas Buah")
    plt.xlabel("Jenis Buah")
    plt.ylabel("Jumlah Data")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribusi_kelas.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="diameter", y="weight", hue="name")
    plt.title("Perbandingan Diameter dan Weight")
    plt.xlabel("Diameter")
    plt.ylabel("Weight")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_diameter_weight.png")
    plt.close()

    numeric_df = df.select_dtypes(include=["number"])
    correlation = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="rocket")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png")
    plt.close()

    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")


def prepare_data(df):
    """Membersihkan data, encoding label, dan membagi train-test."""
    df_clean = df.drop_duplicates().dropna().copy()

    encoder = LabelEncoder()
    df_clean["label"] = encoder.fit_transform(df_clean["name"])

    X = df_clean.drop(columns=["name", "label"])
    y = df_clean["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    split_info = (
        "DATA PREPARATION\n"
        "================\n\n"
        f"Jumlah data setelah dibersihkan: {len(df_clean)}\n"
        f"Jumlah data training: {len(X_train)}\n"
        f"Jumlah data testing: {len(X_test)}\n"
        f"Fitur yang digunakan: {list(X.columns)}\n"
        f"Mapping label: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}\n"
    )

    print(split_info)
    save_text("data_preparation_info.txt", split_info)

    return X_train, X_test, y_train, y_test, encoder


def build_models():
    """Membuat model yang dibandingkan."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)),
            ]
        ),
    }


def safe_filename(name):
    """Mengubah nama model menjadi nama file yang aman."""
    return name.lower().replace(" ", "_")


def plot_confusion_matrix(cm, class_names, model_name):
    """Menyimpan confusion matrix dalam bentuk gambar."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{safe_filename(model_name)}.png")
    plt.close()


def evaluate_models(models, X_train, X_test, y_train, y_test, encoder):
    """Melatih dan mengevaluasi semua model."""
    results = []
    class_names = encoder.classes_

    print("\nMODELING DAN EVALUATION")
    print("=======================")

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names)

        print(f"\nModel: {model_name}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

        report_text = (
            f"MODEL: {model_name}\n"
            f"{'=' * 40}\n\n"
            f"Accuracy : {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall   : {recall:.4f}\n"
            f"F1-Score : {f1:.4f}\n\n"
            f"Confusion Matrix:\n{cm}\n\n"
            f"Classification Report:\n{report}"
        )

        save_text(
            f"classification_report_{safe_filename(model_name)}.txt",
            report_text,
        )

        plot_confusion_matrix(cm, class_names, model_name)

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by="Accuracy",
        ascending=False,
    )

    results_df.to_csv(OUTPUT_DIR / "model_evaluation_metrics.csv", index=False)

    save_text(
        "model_evaluation_metrics.txt",
        results_df.to_string(index=False),
    )

    return results_df


def plot_model_performance(results_df):
    """Membuat grafik perbandingan performa model."""
    melted_df = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric")
    plt.title("Perbandingan Performa Model")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "perbandingan_performa_model.png")
    plt.close()


def main():
    """Menjalankan seluruh tahapan CRISP-DM."""
    setup_output_dir()

    df = load_data(DATA_PATH)

    business_understanding()
    data_understanding(df)
    plot_data_visualization(df)

    X_train, X_test, y_train, y_test, encoder = prepare_data(df)

    models = build_models()
    results_df = evaluate_models(models, X_train, X_test, y_train, y_test, encoder)

    plot_model_performance(results_df)

    print("\nPROGRAM SELESAI")
    print("Semua hasil tersimpan di folder outputs.")
    print("\nRingkasan hasil evaluasi:")
    print(results_df)


if __name__ == "__main__":
    main()