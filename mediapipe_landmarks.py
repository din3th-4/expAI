"""
python script for collecting landmarks on all the original images to see if they 
correlate to the ratings assigned by humans and AI 

"""

import os 
import csv 

import mediapipe as mp 
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


from utilsss import get_img_list, get_dir_label, parse_filenames, get_already_rated, valid_ext

model_path = "face_landmarker.task"
m_name = "mediapipe"
csv_path = f"{m_name}.csv" 

features = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "eyeSquintLeft", 
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawOpen", "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthPucker",
    "noseSneerLeft", "noseSneerRight", "cheekSquintLeft", 
    "cheekSquintRight",
] 

csv_fields = (["img_id", "version", "ambiguous", "r_id", "directory"] 
              + features
            )

def build_landmarker():
    """
    Loads up the FaceLandmarker model once
    """
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)

def extract_features(landmarker, filepath):
    """
    Runs the landmarker on on image and returns feature_scores
    a list of floats 
    """
    image = mp.Image.create_from_file(filepath)
    result = landmarker.detect(image)
    
    if not result.face_blendshapes:
        return None
    
    blendshapes = result.face_blendshapes[0]
    scores = {c.category_name: c.score for c in blendshapes}

    feature_scores = [round(scores.get(name, 0.0), 4) for name in features
                    ]
    
    return feature_scores

def main():

    directory = input("Enter path to image directory: ").strip()

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return
    
    if not os.path.isfile(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    
    dir_label = get_dir_label(directory)

    done = get_already_rated(csv_path, dir_label)
    img_files = get_img_list(directory, done)

    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(csv_fields)

    if not img_files:
        print(f"all images in {dir_label} have already been processed.")
        return
    
    print (f"Processing {len(img_files)} images skipped {len(done)} already processed images"   
        )

    landmarker = build_landmarker()

    for filename in img_files:

        filepath = os.path.join(directory, filename)    

        img_id, version, ambiguous = parse_filenames(filename)

        feature_scores = extract_features(landmarker, filepath)

        if feature_scores is None:
            print(f"No face detected in {filename}. Skipping.")
            continue

        row = [img_id, version, ambiguous, m_name, dir_label] + feature_scores

        with open(csv_path, mode='a', newline='') as f:
            csv.writer(f).writerow(row)

        print(f"{filename} -> done")
 
    landmarker.close()
    print("Done.")
 
 
if __name__ == "__main__":
    main()
