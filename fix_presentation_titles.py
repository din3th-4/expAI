"""Fix template titles and footers in the EXP-AI presentation.

This script edits only repeated title/footer/caption text in the PowerPoint XML.
It creates a corrected copy and leaves the original deck untouched.
"""

from __future__ import annotations

import copy
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


INPUT = Path("presentation/AI_LAB_expai_presentation_ACSAI_2025_2026.pptx")
OUTPUT = Path("presentation/AI_LAB_expai_presentation_ACSAI_2025_2026_titles_fixed.pptx")

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}

ET.register_namespace("a", NS["a"])
ET.register_namespace("p", NS["p"])
ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")

FOOTER = "AI Lab: Human and AI-Based Emotion Recognition Under Image Degradation"

GLOBAL_REPLACEMENTS = {
    "AI Lab: Computer Vision – Signal Processing – Natural Language Processing": FOOTER,
    "Introduction Research Problem Methodology Results Discussion Conclusion": (
        "Introduction Methodology Results Modified Images MediaPipe Conclusion"
    ),
    (
        "ContextProblem and MotivationRelated Work and Gap with Respect to the State of the Art"
        "Proposed Solution and Our ContributionDataset and Experimental Protocol"
        "Critical Discussion and LimitationsConclusion and Future Work"
    ): "Introduction Methodology Results Modified Images MediaPipe Conclusion",
    "Advanced Computer Vision System for Active Surveillance": (
        "Human and AI-Based Emotion Recognition"
    ),
    "From Passive Video Analysis to Intelligent and Adaptive Surveillance": (
        "Methodology: From Images to Emotion Vectors"
    ),
    "Challenges in Real-Time Scene Understanding and Threat Detection": (
        "Building the Emotion Recognition Dataset"
    ),
    "A Vision-Based Pipeline for Risk-Aware Interpretation": (
        "Analysis Pipeline and Contribution"
    ),
    "Evaluating Detection and Anomaly Reasoning in Complex Scenarios": (
        "Results and Experimental Comparison"
    ),
    "Failure Cases, Reproducibility, and Deployment Constraints": (
        "Discussion, Limitations, and Dataset Constraints"
    ),
    "From Visual Monitoring to Intelligent Situation Awareness": (
        "Conclusion and Future Work"
    ),
    "Related Work and Gap with Respect to the State of the Art ●○": "Methodology ●○",
    "Related Work and Gap with Respect to the State of the Art ○●": "Methodology ○●",
    "Problem and Motivation ●○": "Methodology ●○",
    "Problem and Motivation ○●": "Methodology ○●",
    "Proposed Solution and Our Contribution ●○○": "Analysis Approach ●○○",
    "Proposed Solution and Our Contribution ○●○": "Analysis Approach ○●○",
    "Proposed Solution and Our Contribution ○○●": "Analysis Approach ○○●",
    "Dataset and Experimental Protocol ●○○": "Results ●○○",
    "Dataset and Experimental Protocol ○●○": "Results ○●○",
    "Dataset and Experimental Protocol ○○●": "Results ○○●",
    "Critical Discussion and Limitations ●": "Discussion and Limitations ●",
    "Conclusion and Future Work ●": "Conclusion and Future Work ●",
    "Context ●": "Introduction ●",
}

SLIDE_REPLACEMENTS = {
    3: {
        "Advanced Computer Vision System for Active Surveillance": (
            "Facial emotion recognition is now a human-AI comparison"
        ),
    },
    4: {
        "Advanced Computer Vision System for Active Surveillance": (
            "This study compares agreement instead of correctness"
        ),
    },
    6: {
        "From Passive Video Analysis to Intelligent and Adaptive Surveillance": (
            "Methodology Overview"
        ),
    },
    7: {
        "Challenges in Real-Time Scene Understanding and Threat Detection": (
            "Web-selected images define the starting point"
        ),
    },
    8: {
        "Challenges in Real-Time Scene Understanding and Threat Detection": (
            "Image variations test robustness under degradation"
        ),
    },
    9: {
        "Challenges in Real-Time Scene Understanding and Threat Detection": (
            "Human responses were collected through a custom script"
        ),
        "Image Versions": "Human Responses",
    },
    10: {
        "Challenges in Real-Time Scene Understanding and Threat Detection": (
            "FER and DeepFace outputs were collected with Python scripts"
        ),
        "Image Versions": "FER and DeepFace",
    },
    12: {
        "From Passive Video Analysis to Intelligent and Adaptive Surveillance": "Research Problem",
        "Methodology ●○": "Research Problem ●○",
    },
    13: {
        "From Passive Video Analysis to Intelligent and Adaptive Surveillance": "Research Problem",
        "Methodology ○●": "Research Problem ○●",
    },
    15: {
        "A Vision-Based Pipeline for Risk-Aware Interpretation": "Analysis Approach",
    },
    16: {
        "A Vision-Based Pipeline for Risk-Aware Interpretation": "Human-AI Comparison Metrics",
    },
    17: {
        "A Vision-Based Pipeline for Risk-Aware Interpretation": "MediaPipe Feature Association",
    },
    19: {
        "Evaluating Detection and Anomaly Reasoning in Complex Scenarios": (
            "Human-AI Agreement Results"
        ),
    },
    20: {
        "Evaluating Detection and Anomaly Reasoning in Complex Scenarios": (
            "Image Modification Robustness Results"
        ),
    },
    21: {
        "Evaluating Detection and Anomaly Reasoning in Complex Scenarios": (
            "MediaPipe Association Results"
        ),
    },
    23: {
        "Failure Cases, Reproducibility, and Deployment Constraints": (
            "Discussion and Limitations"
        ),
    },
}


def slide_number(name: str) -> int | None:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else None


def shape_text(shape: ET.Element) -> str:
    return "".join(t.text or "" for t in shape.findall(".//a:t", NS)).strip()


def set_shape_text(shape: ET.Element, value: str) -> None:
    text_nodes = shape.findall(".//a:t", NS)
    if not text_nodes:
        return
    text_nodes[0].text = value
    for node in text_nodes[1:]:
        node.text = ""


def fix_slide(xml_bytes: bytes, index: int, total_slides: int) -> bytes:
    root = ET.fromstring(xml_bytes)
    replacements = copy.deepcopy(GLOBAL_REPLACEMENTS)
    replacements = {**replacements, **SLIDE_REPLACEMENTS.get(index, {})}

    for shape in root.findall(".//p:sp", NS):
        text = shape_text(shape)
        if not text:
            continue
        if text == "1/16":
            set_shape_text(shape, f"{index}/{total_slides}")
        elif text in replacements:
            set_shape_text(shape, replacements[text])

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    with zipfile.ZipFile(INPUT, "r") as source:
        names = source.namelist()
        slide_files = [
            name
            for name in names
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        ]
        slide_files = sorted(slide_files, key=lambda name: slide_number(name) or 0)
        total_slides = len(slide_files)

        with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as target:
            for name in names:
                data = source.read(name)
                idx = slide_number(name)
                if name in slide_files and idx is not None:
                    data = fix_slide(data, idx, total_slides)
                target.writestr(name, data)

    print(f"Created {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
