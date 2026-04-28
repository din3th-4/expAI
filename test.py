from deepface import DeepFace 
import numpy as np 
import cv2 

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame1 = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
    framemir = cv2.flip(frame1, 1) 
    
    

    cv2.imshow('webcam', framemir)  

    if cv2.waitKey(1) == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
