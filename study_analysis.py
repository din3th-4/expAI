"""Build reproducible processed datasets for the exploratory emotion study."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utilsss import parse_filenames, valid_ext


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
KEYS = ["img_id", "version", "ambiguous"]
BASE_KEYS = ["img_id", "ambiguous"]
VERSIONS = ["o", "b", "g", "l"]
VERSION_NAMES = {
    "o": "Original",
    "b": "Blur",
    "g": "Greyscale",
    "l": "Low resolution",
}
MODEL_CONFIG = {
    "FER": ("fer_ratings.csv", 1.0),
    "DeepFace": ("deepface.csv", 100.0),
}
METADATA_COLUMNS = [
    "img_id",
    "version",
    "version_name",
    "ambiguous",
    "assigned_category",
]


def parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def validate(data, columns, name):
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def image_manifest(directory):
    rows = []
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(valid_ext):
            continue
        img_id, version, ambiguous = parse_filenames(filename)
        pieces = img_id.split("_", 1)
        rows.append(
            {
                "img_id": img_id,
                "version": version,
                "version_name": VERSION_NAMES.get(version, version),
                "ambiguous": ambiguous,
                "assigned_category": pieces[1] if len(pieces) == 2 else "",
                "filename": filename,
                "image_path": str(Path(directory) / filename),
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(f"No images found in {directory}")
    if manifest.duplicated(KEYS).any():
        raise ValueError("Current image directory contains duplicate study keys")
    return manifest


def normalize_vectors(data):
    result = data.copy()
    result[EMOTIONS] = result[EMOTIONS].apply(
        pd.to_numeric, errors="coerce"
    ).astype(float)
    totals = result[EMOTIONS].sum(axis=1, min_count=1)
    valid = totals.gt(0)
    result.loc[valid, EMOTIONS] = result.loc[valid, EMOTIONS].div(
        totals[valid], axis=0
    )
    result.loc[~valid, EMOTIONS] = np.nan
    return result


def vector_entropy(data):
    values = data[EMOTIONS].to_numpy(dtype=float)
    safe = np.where(values > 0, values, 1.0)
    entropy = -np.nansum(np.where(values > 0, values * np.log2(safe), 0), axis=1)
    entropy /= np.log2(len(EMOTIONS))
    entropy[~np.isfinite(values).any(axis=1)] = np.nan
    return entropy


def add_profile_fields(data):
    result = normalize_vectors(data)
    valid = result[EMOTIONS].notna().any(axis=1)
    result["dominant_emotion"] = pd.Series(
        pd.NA, index=result.index, dtype="object"
    )
    result.loc[valid, "dominant_emotion"] = result.loc[valid, EMOTIONS].idxmax(axis=1)
    result["entropy"] = vector_entropy(result)
    return result


def load_humans(directory, manifest):
    files = sorted(Path(directory).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No participant CSVs found in {directory}")
    raw = pd.concat(
        [pd.read_csv(path).assign(source_file=path.name) for path in files],
        ignore_index=True,
    )
    validate(raw, ["r_id", *KEYS, *EMOTIONS], "Participant data")
    raw["ambiguous"] = raw["ambiguous"].map(parse_bool)
    raw[EMOTIONS] = raw[EMOTIONS].apply(pd.to_numeric, errors="coerce")
    current = raw.merge(manifest[KEYS], on=KEYS, how="inner")

    version = (
        current.groupby(KEYS, as_index=False)
        .agg(
            **{emotion: (emotion, "sum") for emotion in EMOTIONS},
            n_ratings=("r_id", "nunique"),
        )
        .merge(manifest[METADATA_COLUMNS], on=KEYS, how="right")
    )
    version = add_profile_fields(version)
    version["source"] = "Humans (version-specific)"

    pooled = current.groupby(BASE_KEYS, as_index=False).agg(
        **{emotion: (emotion, "sum") for emotion in EMOTIONS},
        n_ratings=("r_id", "nunique"),
        n_responses=("r_id", "size"),
    )
    pooled = add_profile_fields(pooled)
    pooled["source"] = "Human pooled reference"

    pooled_expanded = manifest[METADATA_COLUMNS].merge(
        pooled[BASE_KEYS + EMOTIONS + ["n_ratings", "n_responses", "entropy",
                                      "dominant_emotion"]],
        on=BASE_KEYS,
        how="left",
    )
    pooled_expanded["source"] = "Human pooled reference"
    return raw, current, version, pooled, pooled_expanded


def load_model(name, path, scale, manifest):
    data = pd.read_csv(path)
    validate(data, [*KEYS, *EMOTIONS], name)
    data["ambiguous"] = data["ambiguous"].map(parse_bool)
    data[EMOTIONS] = data[EMOTIONS].apply(pd.to_numeric, errors="coerce") / scale
    stale_count = len(data.merge(manifest[KEYS], on=KEYS, how="left", indicator=True)
                      .query("_merge == 'left_only'"))
    current = data.merge(manifest[METADATA_COLUMNS], on=KEYS, how="inner")
    current = current.groupby(METADATA_COLUMNS, as_index=False)[EMOTIONS].mean()
    current = manifest[METADATA_COLUMNS].merge(
        current, on=METADATA_COLUMNS, how="left"
    )
    current = add_profile_fields(current)
    current["source"] = name
    current["detected"] = current[EMOTIONS].notna().any(axis=1)
    return current, stale_count


def cosine(first, second):
    valid = np.isfinite(first) & np.isfinite(second)
    if not valid.any():
        return np.nan
    denominator = np.linalg.norm(first[valid]) * np.linalg.norm(second[valid])
    return np.dot(first[valid], second[valid]) / denominator if denominator else np.nan


def js_divergence(first, second):
    valid = np.isfinite(first) & np.isfinite(second)
    if not valid.any():
        return np.nan
    p, q = first[valid], second[valid]
    p, q = p / p.sum(), q / q.sum()
    midpoint = (p + q) / 2

    def kl(values, reference):
        mask = values > 0
        return np.sum(values[mask] * np.log2(values[mask] / reference[mask]))

    return (kl(p, midpoint) + kl(q, midpoint)) / 2


def compare_to_humans(reference, models):
    human = reference[KEYS + EMOTIONS + ["dominant_emotion", "entropy"]].rename(
        columns={
            **{emotion: f"human_{emotion}" for emotion in EMOTIONS},
            "dominant_emotion": "human_dominant",
            "entropy": "human_entropy",
        }
    )
    rows = []
    for model in models:
        merged = model.merge(human, on=KEYS, how="left")
        for _, row in merged.iterrows():
            model_vector = row[EMOTIONS].to_numpy(dtype=float)
            human_vector = row[[f"human_{e}" for e in EMOTIONS]].to_numpy(dtype=float)
            valid = np.isfinite(model_vector) & np.isfinite(human_vector)
            rows.append(
                {
                    **{column: row[column] for column in METADATA_COLUMNS},
                    "source": row["source"],
                    "detected": row["detected"],
                    "human_dominant": row["human_dominant"],
                    "model_dominant": row["dominant_emotion"],
                    "dominant_agreement": (
                        row["human_dominant"] == row["dominant_emotion"]
                        if row["detected"] and pd.notna(row["human_dominant"])
                        else np.nan
                    ),
                    "cosine_similarity": cosine(model_vector, human_vector),
                    "mean_absolute_error": (
                        np.mean(np.abs(model_vector[valid] - human_vector[valid]))
                        if valid.any() else np.nan
                    ),
                    "jensen_shannon_divergence": js_divergence(
                        model_vector, human_vector
                    ),
                    "human_entropy": row["human_entropy"],
                    "model_entropy": row["entropy"],
                    "entropy_difference": (
                        row["entropy"] - row["human_entropy"]
                        if pd.notna(row["entropy"]) and pd.notna(row["human_entropy"])
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def modification_robustness(sources):
    rows = []
    for data in sources:
        for base_key, group in data.groupby(BASE_KEYS):
            original = group[group["version"] == "o"]
            if original.empty:
                continue
            original_vector = original.iloc[0][EMOTIONS].to_numpy(dtype=float)
            original_entropy = original.iloc[0]["entropy"]
            for _, row in group[group["version"] != "o"].iterrows():
                modified = row[EMOTIONS].to_numpy(dtype=float)
                valid = np.isfinite(original_vector) & np.isfinite(modified)
                rows.append(
                    {
                        "img_id": base_key[0],
                        "ambiguous": base_key[1],
                        "source": row["source"],
                        "version": row["version"],
                        "version_name": row["version_name"],
                        "cosine_to_original": cosine(modified, original_vector),
                        "mean_absolute_change": (
                            np.mean(np.abs(modified[valid] - original_vector[valid]))
                            if valid.any() else np.nan
                        ),
                        "entropy_change": (
                            row["entropy"] - original_entropy
                            if pd.notna(row["entropy"]) and pd.notna(original_entropy)
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def condition_summary(comparison):
    return (
        comparison.groupby(["source", "ambiguous", "version", "version_name"],
                           dropna=False)
        .agg(
            n_images=("img_id", "size"),
            n_detected=("detected", "sum"),
            mean_cosine_similarity=("cosine_similarity", "mean"),
            mean_absolute_error=("mean_absolute_error", "mean"),
            mean_js_divergence=("jensen_shannon_divergence", "mean"),
            dominant_agreement_rate=("dominant_agreement", "mean"),
            mean_human_entropy=("human_entropy", "mean"),
            mean_model_entropy=("model_entropy", "mean"),
        )
        .reset_index()
    )


def ambiguity_summary(human_pooled, models, comparison):
    frames = []
    human = human_pooled.groupby("ambiguous").agg(
        n_base_images=("img_id", "size"),
        mean_entropy=("entropy", "mean"),
        sd_entropy=("entropy", "std"),
    ).reset_index()
    human["source"] = "Human pooled reference"
    frames.append(human)

    for model in models:
        summary = model.groupby("ambiguous").agg(
            n_base_images=("img_id", "nunique"),
            mean_entropy=("entropy", "mean"),
            sd_entropy=("entropy", "std"),
        ).reset_index()
        summary["source"] = model["source"].iloc[0]
        frames.append(summary)
    result = pd.concat(frames, ignore_index=True)
    agreement = comparison.groupby(["source", "ambiguous"]).agg(
        mean_human_similarity=("cosine_similarity", "mean"),
        dominant_agreement_rate=("dominant_agreement", "mean"),
    ).reset_index()
    return result.merge(agreement, on=["source", "ambiguous"], how="left")


def mediapipe_associations(path, manifest, human_reference, models, comparison):
    if not Path(path).is_file():
        return None, None
    media = pd.read_csv(path)
    validate(media, KEYS, "MediaPipe data")
    media["ambiguous"] = media["ambiguous"].map(parse_bool)
    feature_columns = [
        column for column in media.columns
        if column not in {
            *KEYS, "r_id", "directory", "filename", "face_detected"
        }
    ]
    media[feature_columns] = media[feature_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    media = manifest[METADATA_COLUMNS].merge(media, on=KEYS, how="left")

    targets = []
    for source in [human_reference, *models]:
        item = source[KEYS + EMOTIONS].copy()
        prefix = source["source"].iloc[0].replace(" ", "_").lower()
        item = item.rename(columns={emotion: f"{prefix}_{emotion}" for emotion in EMOTIONS})
        targets.append(item)
    target_data = targets[0]
    for item in targets[1:]:
        target_data = target_data.merge(item, on=KEYS, how="outer")
    disagreement = comparison.pivot_table(
        index=KEYS, columns="source", values="cosine_similarity"
    ).reset_index()
    disagreement = disagreement.rename(
        columns={column: f"{column.lower()}_human_similarity"
                 for column in disagreement.columns if column not in KEYS}
    )
    combined = media.merge(target_data, on=KEYS, how="left").merge(
        disagreement, on=KEYS, how="left"
    )

    target_columns = [
        column for column in combined.columns
        if any(column.endswith(f"_{emotion}") for emotion in EMOTIONS)
        or column.endswith("_human_similarity")
    ]
    rows = []
    for feature in feature_columns:
        for target in target_columns:
            valid = combined[[feature, target]].dropna()
            rows.append(
                {
                    "mediapipe_feature": feature,
                    "target": target,
                    "n": len(valid),
                    "pearson_correlation": (
                        valid[feature].corr(valid[target]) if len(valid) >= 3 else np.nan
                    ),
                }
            )
    return combined, pd.DataFrame(rows)


def make_figures(comparison, robustness, ambiguity, associations, output_dir):
    pass


def main():
    parser = argparse.ArgumentParser(description="Rebuild all study datasets.")
    parser.add_argument("--images", default="Modified_images")
    parser.add_argument("--participants", default="participant_ratings")
    parser.add_argument("--mediapipe", default="mediapipe.csv")
    parser.add_argument("--output", default="study_outputs")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = image_manifest(args.images)
    raw, current, human_version, human_pooled, human_reference = load_humans(
        args.participants, manifest
    )

    models, stale_counts = [], {}
    for name, (path, scale) in MODEL_CONFIG.items():
        if not Path(path).is_file():
            print(f"Warning: missing {name} CSV: {path}")
            continue
        model, stale = load_model(name, path, scale, manifest)
        models.append(model)
        stale_counts[name] = stale
    if not models:
        parser.error("No model CSV data available")

    comparison = compare_to_humans(human_reference, models)
    robustness = modification_robustness([human_version, *models])
    conditions = condition_summary(comparison)
    ambiguity = ambiguity_summary(human_pooled, models, comparison)
    media_combined, associations = mediapipe_associations(
        args.mediapipe, manifest, human_reference, models, comparison
    )

    coverage_rows = [
        {"source": "Image manifest", "rows": len(manifest),
         "unique_keys": len(manifest), "usable_rows": len(manifest), "stale_rows": 0},
        {"source": "Human raw", "rows": len(raw),
         "unique_keys": raw.groupby(KEYS).ngroups, "usable_rows": len(current),
         "stale_rows": len(raw) - len(current)},
    ]
    for model in models:
        coverage_rows.append(
            {"source": model["source"].iloc[0], "rows": len(model),
             "unique_keys": model.groupby(KEYS).ngroups,
             "usable_rows": int(model["detected"].sum()),
             "stale_rows": stale_counts[model["source"].iloc[0]]}
        )
    if media_combined is not None:
        coverage_rows.append(
            {"source": "MediaPipe", "rows": len(media_combined),
             "unique_keys": media_combined.groupby(KEYS).ngroups,
             "usable_rows": int(media_combined.dropna(
                 subset=[c for c in media_combined.columns if c.startswith("brow")],
                 how="all").shape[0]),
             "stale_rows": 0}
        )

    manifest.to_csv(output / "image_manifest.csv", index=False)
    human_version.to_csv(output / "human_version_profiles.csv", index=False)
    human_pooled.to_csv(output / "human_pooled_profiles.csv", index=False)
    pd.concat(models, ignore_index=True).to_csv(
        output / "model_emotion_profiles.csv", index=False
    )
    comparison.to_csv(output / "human_ai_comparison.csv", index=False)
    robustness.to_csv(output / "modification_robustness.csv", index=False)
    conditions.to_csv(output / "condition_summary.csv", index=False)
    ambiguity.to_csv(output / "ambiguity_summary.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "coverage_report.csv", index=False)
    if media_combined is not None:
        media_combined.to_csv(output / "mediapipe_combined_data.csv", index=False)
        associations.to_csv(output / "mediapipe_feature_associations.csv", index=False)

    make_figures(comparison, robustness, ambiguity, associations, output)
    print(f"Rebuilt study datasets and figures in {output.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
