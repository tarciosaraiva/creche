import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=(5,5))

    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, pt1=(x,y), pt2=(x+w,y+h), color=(255,0,0), thickness=2)
        roi_gray = frame_gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=10, minSize=(5,5))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, pt1=(ex,ey), pt2=(ex+ew,ey+eh), color=(0,255,0), thickness=2)

        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=15, minSize=(25,25))
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_color, pt1=(sx,sy), pt2=(sx+sw,sy+sh), color=(0,0,255), thickness=3)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
