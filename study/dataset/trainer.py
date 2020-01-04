import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
script_dir = os.path.dirname(os.path.realpath(__file__))
users_dir = "{0}/users".format(script_dir)

def getImagesAndLabels(path):
    faceSamples = []
    ids = []

    for face_id in os.listdir(path):
        imagePaths = [os.path.join(path, face_id, f) for f in os.listdir(os.path.join(path, face_id))]

        for image in imagePaths:
            PIL_img = Image.open(image).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(int(face_id))

    return faceSamples, ids


print("Training faces...")
faces, ids = getImagesAndLabels(users_dir)
recognizer.train(faces, np.array(ids))
recognizer.write("{0}/trainer/trainer.yml".format(script_dir))
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
