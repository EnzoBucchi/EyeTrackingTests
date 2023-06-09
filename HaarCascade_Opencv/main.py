import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('C:\FACULDADE\EyeTrackingTests\HaarCascade_Opencv\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\FACULDADE\EyeTrackingTests\HaarCascade_Opencv\haarcascade_eye.xml')

run = True
while run:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
    cv2.imshow("Haar Cascade Opencv", frame)
    key = cv2.waitKey(1)
    if key == 27:
        run = False
        
cap.release()
cv2.destroyAllWindows()