"""
Takes the modified_images directory and makes a csv with all the ratings 

"""

from deepface import DeepFace 
import os  
from datetime import datetime
import time 
import csv

from utilsss import emos, csv_fields, get_dir_label, parse_filenames, get_already_rated, get_img_list

model_name = "deepface"
csv_path = f"{model_name}.csv"



def analyze_img(img_path):
    """
    """

    start = time.time()
    result = DeepFace.analyze(img_path, actions=["emotion"], enforce_detection=False)
    elapsed = round(time.time() - start, 3)

    analysis = result[0] if isinstance(result, list) else result
 
    scores = [analysis["emotion"][e] for e in emos]
    dominant = analysis["dominant_emotion"]
 
    return scores, dominant, elapsed

def main():
    directory = input("Enter path to image directory: ").strip()
 
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return
 
    dir_label = get_dir_label(directory)
 
    done = get_already_rated(csv_path, dir_label)
    img_files = get_img_list(directory, done)
 
    if not img_files:
        print("All images in this directory have already been processed by this model.")
        return
 
    print(f"Processing {len(img_files)} images (skipped {len(done)} already done).")
 
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_fields)
 
    for filename in img_files:
        filepath = os.path.join(directory, filename)
        img_id, version, ambiguous = parse_filenames(filename)
 
        try:
            scores, dominant, elapsed = analyze_img(filepath)
        except Exception as e:
            print(f"Skipped {filename} (DeepFace error: {e})")
            continue
 
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
 
        row = [img_id, version, ambiguous, model_name, dir_label] + scores + [dominant, elapsed, timestamp]
 
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
 
        print(f"{filename} -> {dominant}  ({elapsed}s)")
 
    print("Done.")
 
 
if __name__ == "__main__":
    main()