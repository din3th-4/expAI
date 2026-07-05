"""
For collecting the human data  
--------------------
Displays the images form a directory one at a time, for each image, 
then the rater sets a 1-10 slider for each category of emotion - inline with 
the emotion categories from the AI models deepface and FER - the time taken
for the rating will be recorded after each image and all the data will be 
recorded in a unique csv file for each participant 


"""


import os
import time  
from datetime import datetime
import csv 
import sys 

import tkinter as tk 
from tkinter import simpledialog, filedialog, messagebox
from PIL import Image, ImageTk

emos = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

csv_fields = ( ["img_id","version", "ambiguous", "r_id", "directory"] + emos + 
              ["dominant_emotion", "response_time_s", "timestamp"]
            )

output_dir = "participant_ratings"

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


class Ratings:
    """
    class for the participant window instead of loose functions 
    """

    def __init__(self, root, rater_id, directory, img_files, csv_path):

        self.root = root 
        self.rater_id = rater_id 
        self.directory = directory 
        self.img_files = img_files
        self.csv_path = csv_path 

        self.c_index = 0
        self.s_time = None 
        self.tk_img = None 
        self.root.title("Rating")

        self.prog_label = tk.Label(root, text = "", font = ("Arial", 12))
        self.prog_label.pack(pady=(10,0))
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        self.sliders = {}
        slider_frame = tk.Frame(root)
        slider_frame.pack(pady = 10)

        for e in emos:
            row = tk.Frame(slider_frame)
            row.pack(fill = "x", pady=2)
            
            label = tk.Label(row, text= e.capitalize(), width=10, anchor="w")
            label.pack(side="left")

            slider = tk.Scale(row, from_=0, to=10, orient="horizontal", length=300)
            slider.set(0)
            slider.pack(side="left")

            self.sliders[e] = slider
        
        self.next_button = tk.Button(
            root, text="Next", font=("Arial", 12), command = self.on_next
        )
        self.next_button.pack(pady=15)

        if not os.path.isfile(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(csv_fields)
        
        self.load_img(self.c_index)

    def load_img(self, index):

        filename = self.img_files[index]
        filepath = os.path.join(self.directory, filename)

        img = Image.open(filepath)
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_img)

        self.prog_label.config(
            text=f"Image {index+1} of {len(self.img_files)}"
        )

        for slider in self.sliders.values():
            slider.set(0)
        
        self.s_time = time.time()

    def on_next(self):
        """
        Reads the current slider vals, writes a row to the csv, then either loads the next img or
        finishes the session if all images have been rated 
        """

        e_time = time.time()
        time_taken = round(e_time - self.s_time, 3) 

        filename = self.img_files[self.c_index]
        img_id, version, ambiguous = parse_filenames(filename)

        values = [self.sliders[e].get() for e in emos]

        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        dominant_emotion = max(emos, key=lambda e: self.sliders[e].get())

        row = [img_id, version, ambiguous, self.rater_id, self.directory] + values + [dominant_emotion, time_taken, timestamp] 
        
        
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row) 
        
        self.c_index += 1

        if self.c_index >= len(self.img_files):
            self.finish()
        else: self.load_img(self.c_index)
    
    def finish(self):
        """
        clears the screen and turns the nexr button into a close button 
        """

        self.img_label.config(image="")
        self.prog_label.config(text=f"Done, {len(self.img_files)} images rated")

        for slider in self.sliders.values():
            slider.pack_forget()
        
        self.next_button.config(text="Close", command=self.root.destroy)


def b_csv_path(rater_id):
    """
    builds the path to the csv file for a given rater id, and creates the output dir if it doesn't exist
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        p_num = int(rater_id)
        filename = f"ratings_{p_num:03d}.csv"
    except ValueError:
        filename = f"{rater_id}.csv"
    
    return os.path.join(output_dir, filename)


def main():

    rater_id = input ("Enter participant number: ").strip()
    directory = input("Enter path to image directory: ").strip()

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return 

    csv_path = b_csv_path(rater_id)
    rated = get_already_rated(csv_path, directory)
    img_files = get_img_list(directory, rated)

    if not img_files:
        print("All images in this directory have already been rated by this participant.")
        return 
    
    print (f"Loaded {len(img_files)} images for rating. (skipped {len(rated)} already rated images). ")

    root = tk.Tk()
    app = Ratings(root, rater_id, directory, img_files, csv_path)
    root.mainloop()

if __name__ == "__main__":
    main()



        
