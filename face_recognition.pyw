#(C) 2022, HappyAnd_13

import cv2
import xlwt
from xlwt import Workbook
import time
import numpy as np
import os
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
wb = Workbook()

recognizer = cv2.face.LBPHFaceRecognizer_create()
assure_path_exists("trainer/")

recognizer.read('trainer/trainer.yml')

cascadePath = r"haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, im =cam.read()

    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    #faces = faceCascade.detectMultiScale(gray, 1.2,5)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors = 5, minSize = (5, 5)) # Обнаружить лицо

    for(x,y,w,h) in faces:

        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 1)
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if(confidence<70):
            if(Id==1):
                Id="Elena"


            elif(Id==2):
                Id="Fantomas"


            elif(Id==3):
                Id="Ekaterina"
        else:
            Id="Unknown"



        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,0,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 1, (0,255,0), 3)

    cv2.imshow('ANN version 0.9 (c) 2022, HappyAnd_13',im)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()

cv2.destroyAllWindows()
