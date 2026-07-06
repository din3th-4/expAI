"""
python script for collecting landmarks on all the original images to see if they 
correlate to the ratings assigned by humans and AI 

"""

import os 
import sys 
import mediapipe as mp
import cv2
import numpy as np

from utilsss import get_img_list, get_dir_label, parse_filenames, get_already_rated, csv_fields 