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



