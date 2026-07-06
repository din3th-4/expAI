"""Generate the complete graph and data-diagram pack for the emotion study.

Run after ``study_analysis.py`` has rebuilt the CSV files in ``study_outputs``.
The figures are saved to ``study_outputs/graphs``.
"""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("study_outputs")
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
    "Human pooled reference": "#4e79a7",
    "Humans (version-specific)": "#4e79a7",
    "FER": "#f28e2b",
    "DeepFace": "#59a14f",
    "MediaPipe": "#b07aa1",
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
        color="#e15759",
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
    fig, axis = plt.subplots(figsize=(13, 4.5))
    axis.axis("off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    boxes = [
        ("Web-selected\nimage set", "Researcher-assigned categories,\nnot certified ground truth"),
        ("Image versions", "Original, blur, greyscale,\nlow resolution"),
        ("Human ratings", "Seven-emotion participant\nscore vectors"),
        ("AI emotion models", "FER + DeepFace\nseven-emotion vectors"),
        ("MediaPipe", "Facial blendshape\nfeature vectors"),
        ("Exploratory analysis", "Agreement, ambiguity,\nrobustness, associations"),
    ]
    xs = np.linspace(0.08, 0.92, len(boxes))
    for i, (title, body) in enumerate(boxes):
        x = xs[i]
        rect = plt.Rectangle(
            (x - 0.068, 0.44), 0.136, 0.34,
            facecolor="#f4f6f8", edgecolor="#4e79a7", linewidth=1.5
        )
        axis.add_patch(rect)
        axis.text(x, 0.68, title, ha="center", va="center",
                  fontsize=10, weight="bold")
        axis.text(
            x,
            0.54,
            "\n".join(textwrap.wrap(body.replace("\n", " "), width=24)),
            ha="center",
            va="center",
            fontsize=8,
        )
        if i < len(boxes) - 1:
            axis.annotate(
                "",
                xy=(xs[i + 1] - 0.078, 0.61),
                xytext=(x + 0.078, 0.61),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#555"},
            )

    counts = ", ".join(
        f"{row.source}: {row.usable_rows}/{row.rows}"
        for row in coverage.itertuples()
    )
    axis.text(
        0.5, 0.2,
        "Current coverage: " + counts,
        ha="center", va="center", fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#fff7e6", "ec": "#f28e2b"},
    )
    axis.set_title("Study data pipeline diagram", fontsize=14, weight="bold")
    return savefig("02_study_pipeline_diagram.png")


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
    data = human.sort_values(["ambiguous", "img_id"]).set_index("img_id")
    matrix = data[EMOTIONS].to_numpy()
    fig, axis = plt.subplots(figsize=(10, max(6, len(data) * 0.22)))
    im = axis.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(EMOTIONS)), EMOTIONS, rotation=35, ha="right")
    labels = [
        f"{idx}{' (A)' if amb else ''}"
        for idx, amb in zip(data.index, data["ambiguous"])
    ]
    axis.set_yticks(np.arange(len(data)), labels, fontsize=8)
    axis.set_title("Human pooled emotion profiles by base image",
                   fontsize=13, weight="bold", pad=12)
    fig.colorbar(im, ax=axis, label="Normalized human score")
    return savefig("04_human_profile_heatmap.png")


def plot_similarity_boxplots(comparison):
    metric = "cosine_similarity"
    data = comparison[comparison["detected"].fillna(True)].dropna(subset=[metric])
    groups = []
    labels = []
    colors = []
    for source in ["FER", "DeepFace"]:
        for version in VERSION_ORDER:
            values = data[(data["source"] == source) & (data["version"] == version)][metric]
            groups.append(values)
            labels.append(f"{source}\n{VERSION_NAMES[version]}")
            colors.append(SOURCE_COLORS[source])

    fig, axis = plt.subplots(figsize=(12, 5))
    bp = axis.boxplot(groups, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axis.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=30, ha="right")
    axis.set_ylim(0, 1.05)
    format_axes(axis, "Human-AI cosine similarity by model and image version",
                ylabel="Cosine similarity")
    return savefig("05_similarity_distributions.png")


def plot_condition_metrics(conditions):
    metrics = [
        ("mean_cosine_similarity", "Mean cosine similarity", "Higher = closer"),
        ("mean_absolute_error", "Mean absolute error", "Lower = closer"),
        ("dominant_agreement_rate", "Dominant-emotion agreement", "Higher = more same labels"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=False)
    for axis, (metric, title, ylabel) in zip(axes, metrics):
        pivot = conditions.pivot_table(
            index=["version_name", "ambiguous"],
            columns="source",
            values=metric,
            aggfunc="mean",
        )
        labels = [
            f"{version}\n{'Ambiguous' if amb else 'Non-ambiguous'}"
            for version, amb in pivot.index
        ]
        x = np.arange(len(pivot))
        width = 0.35
        for i, source in enumerate(["FER", "DeepFace"]):
            if source in pivot:
                axis.bar(
                    x + (i - 0.5) * width,
                    pivot[source],
                    width,
                    label=source,
                    color=SOURCE_COLORS[source],
                )
        axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        format_axes(axis, title, ylabel=ylabel)
        if metric != "mean_absolute_error":
            axis.set_ylim(0, 1)
    axes[0].legend(frameon=False)
    return savefig("06_condition_metric_panels.png")


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
                 label="Non-ambiguous", color="#4e79a7")
        axis.bar(x + width / 2, pivot.get(True, np.nan), width,
                 label="Ambiguous", color="#e15759")
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
    data = robustness.dropna(subset=["cosine_to_original", "mean_absolute_change"])
    fig, axis = plt.subplots(figsize=(8, 6))
    for source, group in data.groupby("source"):
        axis.scatter(
            group["mean_absolute_change"],
            group["cosine_to_original"],
            s=42,
            alpha=0.75,
            label=source,
            color=SOURCE_COLORS.get(source),
        )
    axis.set_ylim(0, 1.05)
    format_axes(
        axis,
        "Robustness tradeoff: vector change vs similarity to original",
        xlabel="Mean absolute change from original",
        ylabel="Cosine similarity to original",
    )
    axis.legend(frameon=False)
    return savefig("09_robustness_scatter.png")


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
    data = comparison[comparison["detected"].fillna(True)]
    pivot = data.pivot_table(
        index="assigned_category",
        columns="source",
        values="dominant_agreement",
        aggfunc="mean",
    ).reindex(EMOTIONS)
    fig, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot))
    width = 0.35
    for i, source in enumerate(["FER", "DeepFace"]):
        axis.bar(x + (i - 0.5) * width, pivot[source], width,
                 label=source, color=SOURCE_COLORS[source])
    axis.set_xticks(x, pivot.index, rotation=30, ha="right")
    axis.set_ylim(0, 1)
    format_axes(axis, "Dominant agreement by researcher-assigned category",
                ylabel="Human-model dominant agreement rate")
    axis.legend(frameon=False)
    return savefig("11_agreement_by_assigned_category.png")


def plot_entropy_scatter(comparison):
    data = comparison[comparison["detected"].fillna(True)].dropna(
        subset=["human_entropy", "model_entropy"]
    )
    fig, axis = plt.subplots(figsize=(7, 6))
    for source, group in data.groupby("source"):
        axis.scatter(group["human_entropy"], group["model_entropy"],
                     s=45, alpha=0.75, label=source,
                     color=SOURCE_COLORS.get(source))
    high = max(data["human_entropy"].max(), data["model_entropy"].max()) * 1.05
    axis.plot([0, high], [0, high], color="#777", linestyle="--", linewidth=1)
    axis.set_xlim(0, high)
    axis.set_ylim(0, high)
    format_axes(axis, "Human vs model entropy",
                xlabel="Human entropy", ylabel="Model entropy")
    axis.legend(frameon=False)
    return savefig("12_entropy_scatter.png")


def plot_mediapipe_top_associations(associations):
    data = associations.dropna(subset=["pearson_correlation"]).copy()
    data["absolute"] = data["pearson_correlation"].abs()
    top = data.nlargest(30, "absolute").sort_values("pearson_correlation")
    fig, axis = plt.subplots(figsize=(12, 9))
    labels = top["mediapipe_feature"] + " → " + top["target"]
    axis.barh(
        labels,
        top["pearson_correlation"],
        color=np.where(top["pearson_correlation"] >= 0, "#4e79a7", "#e15759"),
    )
    axis.axvline(0, color="#333", linewidth=0.8)
    format_axes(axis, "Top MediaPipe feature associations",
                xlabel="Pearson correlation")
    return savefig("13_mediapipe_top_associations.png")


def plot_mediapipe_heatmap(associations):
    data = associations.dropna(subset=["pearson_correlation"]).copy()
    keep_targets = [
        target for target in data["target"].unique()
        if target.startswith("human_pooled_reference_")
        or target.startswith("fer_")
        or target.startswith("deepface_")
    ]
    data = data[data["target"].isin(keep_targets)]
    data["absolute"] = data["pearson_correlation"].abs()
    top_features = (
        data.groupby("mediapipe_feature")["absolute"]
        .max()
        .nlargest(14)
        .index
    )
    short_targets = []
    for target in keep_targets:
        readable = (
            target.replace("human_pooled_reference_", "human_")
            .replace("deepface_", "df_")
            .replace("fer_", "fer_")
        )
        short_targets.append(readable)
    mapping = dict(zip(keep_targets, short_targets))
    matrix = data[data["mediapipe_feature"].isin(top_features)].pivot_table(
        index="mediapipe_feature",
        columns="target",
        values="pearson_correlation",
        aggfunc="mean",
    )
    matrix = matrix.reindex(top_features)
    matrix = matrix.rename(columns=mapping)
    fig, axis = plt.subplots(figsize=(14, 8))
    im = axis.imshow(matrix.fillna(0), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
    axis.set_title("MediaPipe feature correlation heatmap", fontsize=13, weight="bold", pad=12)
    fig.colorbar(im, ax=axis, label="Pearson correlation")
    return savefig("14_mediapipe_correlation_heatmap.png")


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
        plot_mediapipe_heatmap(associations),
        plot_key_mediapipe_features(media),
        plot_data_relationship_diagram(),
    ]
    index = write_index(paths)
    print(f"Generated {len(paths)} graphs in {GRAPH_DIR.resolve()}")
    print(f"Index: {index.resolve()}")


if __name__ == "__main__":
    main()
