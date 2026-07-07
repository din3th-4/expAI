# AI Emotion Recognition Study

This project compares human emotion perception with two AI-based facial emotion recognition models: **FER** and **DeepFace**. It also uses **MediaPipe** facial landmarks/blendshapes to help explain which facial features are connected to human and AI emotion scores.

The dataset was collected from web-selected images, so the assigned emotion labels are not treated as certified ground truth. Because of that, the project focuses on **agreement, disagreement, robustness, and interpretation**, rather than simply saying which model is right or wrong.

## Main Research Idea

The study asks how humans, FER, and DeepFace interpret facial emotions under different image conditions.

The project compares:

- human participant ratings
- FER emotion scores
- DeepFace emotion scores
- MediaPipe facial feature data
- original and modified image versions

The main goal is to understand where humans and AI models agree, where they disagree, and how image changes such as blur, greyscale, and low resolution affect emotion interpretation.

## Project Pipeline

Run the project in this order.

### 1. Prepare the image dataset

Place the original web-selected images in the correct image folder.

The assigned labels are used as categories for analysis, but they should not be treated as absolute truth because the dataset is not from a certified emotion database.

Some images may be intentionally ambiguous so the study can examine uncertain emotion interpretation.

### 2. Run image modification first

Before running FER, DeepFace, MediaPipe, or the final analysis, create the modified image versions.

Run:

```bash
python image_edit_script.py
```

This creates image variants such as:

- original
- greyscale
- blurred
- low resolution

These variants are used to test how stable human and AI emotion recognition is when image quality changes.

### 3. Run FER

Run FER on the modified image dataset:

```bash
python fer_script.py
```

Expected output:

```text
fer_ratings.csv
```

This file stores FER emotion scores for each image.

### 4. Run DeepFace

Run DeepFace on the same modified image dataset:

```bash
python df_script.py
```

Expected output:

```text
deepface.csv
```

This file stores DeepFace emotion scores for each image.

### 5. Collect human participant responses

Human participant ratings are collected with:

```bash
python hum_script.py
```

Participant outputs are saved in:

```text
participant_ratings/
```

Each participant file stores their emotion ratings for the images. These responses are later pooled into human emotion profiles.

### 6. Run MediaPipe

Run the MediaPipe feature extraction script:

```bash
python mediapipe_landmarks.py
```

Expected output:

```text
mediapipe.csv
```

MediaPipe is not used as an emotion classifier in this study. Instead, it extracts facial landmark/blendshape data that can help explain emotion judgments.

For example:

- widened eyes may relate to surprise
- smiling features may relate to happiness
- jaw opening may relate to surprise or strong expression

### 7. Run the main study analysis

After FER, DeepFace, human ratings, and MediaPipe outputs are ready, run:

```bash
python study_analysis.py
```

This combines the data and creates analysis outputs in:

```text
study_outputs/
```

Important output files include:

```text
study_outputs/coverage_report.csv
study_outputs/human_pooled_profiles.csv
study_outputs/model_emotion_profiles.csv
study_outputs/human_ai_comparison.csv
study_outputs/modification_robustness.csv
study_outputs/ambiguity_summary.csv
study_outputs/mediapipe_combined_data.csv
study_outputs/mediapipe_feature_associations.csv
```

### 8. Generate graphs

To create graphs and visual diagrams, run:

```bash
python make_study_graphs.py
```

The graphs are saved inside the study outputs folder and can be used in the report or presentation.

## Recommended Full Run Order

```text
1. Add original images
2. python image_edit_script.py
3. python fer_script.py
4. python df_script.py
5. python hum_script.py
6. python mediapipe_landmarks.py
7. python study_analysis.py
8. python make_study_graphs.py
```

## How to Discuss the Results

Because the dataset is web-selected and not certified, the study should avoid claiming that a model or human participant is simply correct or incorrect.

Instead, discuss:

- where humans and AI models agree
- where humans and AI models disagree
- which emotions are unstable or ambiguous
- how image modifications affect emotion interpretation
- whether FER or DeepFace is more similar to pooled human responses
- whether MediaPipe features help explain the emotion scores

Example findings from the project:

- FER aligned more closely with pooled human ratings than DeepFace.
- DeepFace was strongly affected by blur.
- AI models struggled with disgust.
- Greyscale images shifted some human responses toward sadness.
- Ambiguous images produced mixed human responses and lower agreement.
- MediaPipe helped connect emotion scores to visible facial movements.

## Notes for Future Work

- If new images are added, rerun the pipeline from the image modification step.
- If new participant ratings are added, rerun `study_analysis.py` and `make_study_graphs.py`.
- If MediaPipe features change, rerun `mediapipe_landmarks.py`, then rerun the analysis and graphs.
- Keep image names consistent across all scripts and CSV files.
- Do not edit another person's MediaPipe script unless they are responsible for that part and approve the change.