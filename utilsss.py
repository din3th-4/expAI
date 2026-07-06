
import os 
import os
import time  
from datetime import datetime
import csv 
import sys 


emos = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

csv_fields = ( ["img_id","version", "ambiguous", "r_id", "directory"] + emos + 
              ["dominant_emotion", "response_time_s", "timestamp"]
            )

def parse_filenames(filename):
    """
    Pulls img_id, version, and ambiguous from the filename, which is expected to be in the format:
    number_emotion_version_a.png    ex: 1_happy_o.png
                                        1_happy_o_a.png -> ambiguous img

    returns: (img_id, version, ambiguous(true/false))
    img_id -> 'number_emotion' ex: 1_happy
    version -> single letter version ex: o, b, g, l

    """

    n = os.path.splitext(filename)[0]
    p = n.split("_")

    ambiguous = p[-1] == "a"
    if ambiguous:
        version = p[-2]
        img_id = "_".join(p[:-2])
    else:
        version = p[-1]
        img_id = "_".join(p[:-1])
    
    return img_id, version, ambiguous

def get_dir_label(directory):
    """
    builds a short label to keep in the csv 'directory' column: parent_folder/folder_name
    if the path only has one folder level, returns just the folder name
    """

    name = os.path.normpath(directory)
    folder = os.path.basename(name)
    parent = os.path.basename(os.path.dirname(name))    

    if parent:
        return f"{parent}/{folder}"
    
    return folder



def get_already_rated(csv_path, directory): 
    """
    check for images that have been rated already by a subject, and return a set with the ids
    of the already rated images, this is helpful in case of an interrupted session with a subject 
    """
    rated = set()
    if not os.path.isfile(csv_path):
        return rated 
    with open(csv_path, "r", newline= "") as f:
        reader = csv.DictReader(f)
        for row  in reader:
            if row["directory"] == directory:
                rated.add((row["img_id"], row["version"]))
    
    return rated 

def get_img_list(directory, rated):
    """
    Scan directory for image files, and get them into a list,
    exlcuding the already sorted ones  
    """
    files = sorted(
        f for f in os.listdir(directory) 
        if f.lower().endswith(valid_ext)
    )
    rem = []
    for f in files:
        img_id, version, ambiguous = parse_filenames(f)
        if (img_id, version) not in rated:
            rem.append(f)

    return rem