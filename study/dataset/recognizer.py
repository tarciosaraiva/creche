import cv2
import os
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
trainer_file = "{0}/trainer/trainer.yml".format(script_dir)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_file)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

names = ['None', 'Tarcio', 'Fabiana']

cam = cv2.VideoCapture(0)
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img, pt1=(x,y), pt2=(x+w,y+h), color=(255,0,0), thickness=2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        print(id, confidence)
        confidence_str = "  {0}%".format(round(100 - confidence))

        if (confidence < 100):
            id_str = names[id]
        else:
            id_str = "unknown"

        cv2.putText(img, id_str, (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, confidence_str, (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
