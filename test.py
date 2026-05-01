from deepface import DeepFace 
import numpy as np 
import cv2 

cap = cv2.VideoCapture(0)

frame_count = 0 
frame_rate = 35

last_emotion = "analysing..."

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
    frame = cv2.flip(frame, 1) 

    if frame_count % frame_rate == 0:
        try:
            result = DeepFace.analyze(frame, actions= ['emotion'], enforce_detection= False) 
            last_emotion = result[0]['dominant_emotion']
        except Exception as e:
            print(e)
    
    frame_count += 1 
    
    cv2.putText(frame, last_emotion, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow('webcam', frame)  

    

    if cv2.waitKey(1) == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()
