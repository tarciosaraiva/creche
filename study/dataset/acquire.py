import cv2
import os
import pathlib
import uuid

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face_id = input('enter person id and press enter ==> ')

script_dir = os.path.dirname(os.path.realpath(__file__))
face_dir = "{0}/users/{1}".format(script_dir, face_id)
pathlib.Path(face_dir).mkdir(parents=True, exist_ok=True)

print("saving faces at ", face_dir)

count = 0

while True:
    ret, img = cam.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img, pt1=(x,y), pt2=(x+w,y+h), color=(255,0,0), thickness=2)
        count += 1
        cv2.imwrite("{0}/{1}.jpg".format(face_dir, uuid.uuid4()), gray_img[y:y+h,x:x+w])

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or count >= 300:
        break

cam.release()
cv2.destroyAllWindows()
