from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("study_outputs")
RED = "#e15759"
BLUE = "#4e79a7"
ORANGE = "#f28e2b"
GREEN = "#59a14f"
PINK = "#b07aa1"
GRAPH_DIR = OUTPUT_DIR / "graphs"
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
VERSION_ORDER = ["o", "b", "g", "l"]
VERSION_NAMES = {
    "o": "Original",
    "b": "Blur",
    "g": "Greyscale",
    "l": "Low resolution",
}
SOURCE_COLORS = {
    "Human pooled reference": BLUE,
    "Humans (version-specific)": BLUE,
    "FER": ORANGE,
    "DeepFace": GREEN,
    "MediaPipe": PINK,
}


def read_csv(name):
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def savefig(name):
    path = GRAPH_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def format_axes(axis, title, xlabel=None, ylabel=None):
    axis.set_title(title, fontsize=13, weight="bold", pad=12)
    if xlabel:
        axis.set_xlabel(xlabel)
    if ylabel:
        axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_coverage(coverage):
    data = coverage.copy()
    data["usable_rows"] = pd.to_numeric(data["usable_rows"], errors="coerce")
    data["rows"] = pd.to_numeric(data["rows"], errors="coerce")
    data["missing_or_stale"] = data["rows"] - data["usable_rows"]

    fig, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(data))
    axis.bar(x, data["usable_rows"], color="#4e79a7", label="Usable/current rows")
    axis.bar(
        x,
        data["missing_or_stale"],
        bottom=data["usable_rows"],
        color=RED,
        label="Stale or unusable rows",
    )
    axis.set_xticks(x, data["source"], rotation=25, ha="right")
    for i, row in data.iterrows():
        axis.text(i, row["rows"] + 2, f"{int(row['usable_rows'])}/{int(row['rows'])}",
                  ha="center", fontsize=9)
    format_axes(axis, "Data coverage by source", ylabel="Rows")
    axis.legend(frameon=False)
    return savefig("01_data_coverage.png")


def plot_study_pipeline(coverage):
    pass


def plot_mean_emotion_profiles(human, models):
    frames = []
    h = human.copy()
    h["source"] = "Human pooled reference"
    frames.append(h[["source", *EMOTIONS]])
    frames.append(models[models["detected"].fillna(True)][["source", *EMOTIONS]])
    data = pd.concat(frames, ignore_index=True)
    mean_profiles = data.groupby("source")[EMOTIONS].mean().reindex(
        ["Human pooled reference", "FER", "DeepFace"]
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(EMOTIONS))
    width = 0.25
    for i, source in enumerate(mean_profiles.index):
        axis.bar(
            x + (i - 1) * width,
            mean_profiles.loc[source],
            width,
            label=source,
            color=SOURCE_COLORS.get(source),
        )
    axis.set_xticks(x, EMOTIONS)
    axis.set_ylim(0, max(0.5, mean_profiles.max().max() * 1.2))
    format_axes(axis, "Average emotion distribution by source",
                ylabel="Mean normalized score")
    axis.legend(frameon=False)
    return savefig("03_average_emotion_profiles.png")


def plot_human_heatmap(human):
    pass


def plot_similarity_boxplots(comparison):
    pass


def plot_condition_metrics(conditions):
    pass


def plot_ambiguity_summary(ambiguity):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [
        ("mean_entropy", "Mean response entropy", "Entropy"),
        ("mean_human_similarity", "Mean human-model similarity", "Cosine similarity"),
        ("dominant_agreement_rate", "Dominant agreement", "Agreement rate"),
    ]
    for axis, (metric, title, ylabel) in zip(axes, metrics):
        data = ambiguity.dropna(subset=[metric])
        pivot = data.pivot_table(
            index="source", columns="ambiguous", values=metric, aggfunc="mean"
        )
        x = np.arange(len(pivot))
        width = 0.35
        axis.bar(x - width / 2, pivot.get(False, np.nan), width,
                 label="Non-ambiguous", color=BLUE)
        axis.bar(x + width / 2, pivot.get(True, np.nan), width,
                 label="Ambiguous", color=RED)
        axis.set_xticks(x, pivot.index, rotation=20, ha="right")
        format_axes(axis, title, ylabel=ylabel)
        if metric != "mean_entropy":
            axis.set_ylim(0, 1)
    axes[0].legend(frameon=False)
    return savefig("07_ambiguity_panels.png")


def plot_robustness(robustness):
    data = robustness.dropna(subset=["cosine_to_original"]).copy()
    pivot = data.pivot_table(
        index="version_name", columns="source", values="cosine_to_original", aggfunc="mean"
    ).reindex(["Blur", "Greyscale", "Low resolution"])
    fig, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot))
    width = 0.25
    for i, source in enumerate(["Humans (version-specific)", "FER", "DeepFace"]):
        if source in pivot:
            axis.bar(
                x + (i - 1) * width,
                pivot[source],
                width,
                label=source,
                color=SOURCE_COLORS.get(source),
            )
    axis.set_xticks(x, pivot.index)
    axis.set_ylim(0, 1.05)
    format_axes(axis, "Modification robustness: similarity to original",
                ylabel="Cosine similarity to original")
    axis.legend(frameon=False)
    return savefig("08_modification_robustness_full.png")


def plot_robustness_scatter(robustness):
    pass


def plot_confusion_matrices(comparison):
    data = comparison[
        comparison["detected"].fillna(True)
        & comparison["human_dominant"].notna()
        & comparison["model_dominant"].notna()
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, source in zip(axes, ["FER", "DeepFace"]):
        group = data[data["source"] == source]
        matrix = pd.crosstab(
            group["human_dominant"],
            group["model_dominant"],
        ).reindex(index=EMOTIONS, columns=EMOTIONS, fill_value=0)
        im = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(np.arange(len(EMOTIONS)), EMOTIONS, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(EMOTIONS)), EMOTIONS)
        axis.set_xlabel(f"{source} dominant emotion")
        axis.set_ylabel("Human dominant emotion")
        axis.set_title(f"Human vs {source} dominant-emotion matrix", weight="bold")
        for i in range(len(EMOTIONS)):
            for j in range(len(EMOTIONS)):
                val = int(matrix.iloc[i, j])
                if val:
                    axis.text(j, i, val, ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    return savefig("10_dominant_emotion_matrices.png")


def plot_assigned_category_agreement(comparison):
    pass


def plot_entropy_scatter(comparison):
    pass


def plot_mediapipe_top_associations_by_model(associations):
    model_specs = [
        (
            "human_pooled_reference_",
            "Human pooled",
            "13a_mediapipe_top_associations_human_pooled.png",
            BLUE,
        ),
        (
            "fer_",
            "FER",
            "13b_mediapipe_top_associations_fer.png",
            ORANGE,
        ),
        (
            "deepface_",
            "DeepFace",
            "13c_mediapipe_top_associations_deepface.png",
            GREEN,
        ),
    ]
    paths = []
    clean = associations.dropna(subset=["pearson_correlation"]).copy()
    clean["absolute"] = clean["pearson_correlation"].abs()

    for prefix, title, filename, positive_color in model_specs:
        data = clean[clean["target"].str.startswith(prefix)].copy()
        if data.empty:
            continue
        top = data.nlargest(25, "absolute").sort_values("pearson_correlation")
        labels = (
            top["mediapipe_feature"]
            + " → "
            + top["target"].str.replace(prefix, "", regex=False)
        )

        fig, axis = plt.subplots(figsize=(11, 8))
        axis.barh(
            labels,
            top["pearson_correlation"],
            color=np.where(top["pearson_correlation"] >= 0, positive_color, RED),
        )
        axis.axvline(0, color="#333", linewidth=0.8)
        format_axes(
            axis,
            f"Top MediaPipe associations with {title} emotion scores",
            xlabel="Pearson correlation",
        )
        paths.append(savefig(filename))
    return paths


def plot_mediapipe_heatmap(associations):
    pass


def plot_key_mediapipe_features(media):
    key_features = [
        "eyeWideLeft",
        "eyeWideRight",
        "jawOpen",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthFrownLeft",
        "mouthFrownRight",
    ]
    available = [feature for feature in key_features if feature in media.columns]
    data = media.copy()
    data["version_name"] = pd.Categorical(
        data["version_name"],
        categories=["Original", "Blur", "Greyscale", "Low resolution"],
        ordered=True,
    )
    means = data.groupby("version_name", observed=False)[available].mean()
    fig, axis = plt.subplots(figsize=(11, 5))
    for feature in available:
        axis.plot(means.index.astype(str), means[feature],
                  marker="o", linewidth=2, label=feature)
    format_axes(axis, "Key MediaPipe feature means by image version",
                ylabel="Mean blendshape score")
    axis.legend(frameon=False, ncol=2)
    return savefig("15_mediapipe_feature_version_lines.png")


def plot_data_relationship_diagram():
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.axis("off")
    nodes = {
        "images": (0.12, 0.72, "Image manifest\n140 image-version rows"),
        "human": (0.38, 0.86, "Human ratings\nraw participant scores"),
        "models": (0.38, 0.58, "FER + DeepFace\nemotion vectors"),
        "media": (0.38, 0.30, "MediaPipe\nblendshape features"),
        "profiles": (0.66, 0.72, "Normalized profiles\n7 emotion dimensions"),
        "analysis": (0.88, 0.52, "Outputs\nagreement, ambiguity,\nrobustness, association"),
    }
    for key, (x, y, label) in nodes.items():
        color = "#f4f6f8"
        edge = "#4e79a7" if key != "analysis" else "#f28e2b"
        rect = plt.Rectangle((x - 0.105, y - 0.08), 0.21, 0.16,
                             facecolor=color, edgecolor=edge, linewidth=1.5)
        axis.add_patch(rect)
        axis.text(x, y, label, ha="center", va="center", fontsize=10, weight="bold")
    arrows = [
        ("images", "human"), ("images", "models"), ("images", "media"),
        ("human", "profiles"), ("models", "profiles"),
        ("profiles", "analysis"), ("media", "analysis"),
    ]
    for start, end in arrows:
        x1, y1, _ = nodes[start]
        x2, y2, _ = nodes[end]
        axis.annotate(
            "",
            xy=(x2 - 0.12 if x2 > x1 else x2 + 0.12, y2),
            xytext=(x1 + 0.12 if x2 > x1 else x1 - 0.12, y1),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#555"},
        )
    axis.set_title("Data relationship diagram", fontsize=14, weight="bold")
    return savefig("16_data_relationship_diagram.png")


def write_index(paths):
    index = GRAPH_DIR / "README.md"
    lines = [
        "# Study graph pack",
        "",
        "Generated from the current CSV files in `study_outputs`.",
        "",
    ]
    descriptions = {
        "01_data_coverage.png": "Checks which data sources are complete/current.",
        "02_study_pipeline_diagram.png": "Shows the study flow from images to outputs.",
        "03_average_emotion_profiles.png": "Compares average seven-emotion distributions.",
        "04_human_profile_heatmap.png": "Shows participant consensus per base image.",
        "05_similarity_distributions.png": "Shows human-AI similarity spread by version.",
        "06_condition_metric_panels.png": "Summarizes cosine, MAE, and dominant agreement.",
        "07_ambiguity_panels.png": "Compares ambiguous vs non-ambiguous results.",
        "08_modification_robustness_full.png": "Shows robustness to blur/greyscale/low-res.",
        "09_robustness_scatter.png": "Shows change magnitude versus similarity.",
        "10_dominant_emotion_matrices.png": "Human-model dominant-emotion matrices.",
        "11_agreement_by_assigned_category.png": "Agreement stratified by intended category.",
        "12_entropy_scatter.png": "Human uncertainty versus model uncertainty.",
        "13_mediapipe_top_associations.png": "Strongest MediaPipe correlations.",
        "13a_mediapipe_top_associations_human_pooled.png": "Strongest MediaPipe correlations for human pooled emotion scores.",
        "13b_mediapipe_top_associations_fer.png": "Strongest MediaPipe correlations for FER emotion scores.",
        "13c_mediapipe_top_associations_deepface.png": "Strongest MediaPipe correlations for DeepFace emotion scores.",
        "14_mediapipe_correlation_heatmap.png": "MediaPipe feature-target correlation map.",
        "15_mediapipe_feature_version_lines.png": "Key MediaPipe feature means by version.",
        "16_data_relationship_diagram.png": "High-level data table relationship diagram.",
    }
    for path in paths:
        lines.append(f"- `{path.name}` — {descriptions.get(path.name, '')}")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
    })

    coverage = read_csv("coverage_report.csv")
    human = read_csv("human_pooled_profiles.csv")
    models = read_csv("model_emotion_profiles.csv")
    comparison = read_csv("human_ai_comparison.csv")
    robustness = read_csv("modification_robustness.csv")
    ambiguity = read_csv("ambiguity_summary.csv")
    associations = read_csv("mediapipe_feature_associations.csv")
    media = read_csv("mediapipe_combined_data.csv")

    for data in [human, models, comparison, robustness, ambiguity, media]:
        if "ambiguous" in data.columns:
            data["ambiguous"] = data["ambiguous"].astype(str).str.lower().isin(["true", "1"])
    for data in [models, comparison]:
        if "detected" in data.columns:
            data["detected"] = data["detected"].astype(str).str.lower().isin(["true", "1"])

    paths = [
        plot_coverage(coverage),
        plot_study_pipeline(coverage),
        plot_mean_emotion_profiles(human, models),
        plot_human_heatmap(human),
        plot_similarity_boxplots(comparison),
        plot_condition_metrics(read_csv("condition_summary.csv")),
        plot_ambiguity_summary(ambiguity),
        plot_robustness(robustness),
        plot_robustness_scatter(robustness),
        plot_confusion_matrices(comparison),
        plot_assigned_category_agreement(comparison),
        plot_entropy_scatter(comparison),
        plot_mediapipe_top_associations(associations),
        *plot_mediapipe_top_associations_by_model(associations),
        plot_mediapipe_heatmap(associations),
        plot_key_mediapipe_features(media),
        plot_data_relationship_diagram(),
    ]
    index = write_index(paths)
    print(f"Generated {len(paths)} graphs in {GRAPH_DIR.resolve()}")
    print(f"Index: {index.resolve()}")


if __name__ == "__main__":
    main()
