import cv2 as cv
import numpy as np
from PIL import Image
import os
 
def draw_boundary(img, clf):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_detect = face_cascade.detectMultiScale(gray_img, 1.1, 5)
    xywh = []
    for (x, y, w, h) in face_detect:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        cv.rectangle(img, (x, y-50), (x+w, y), (0, 0, 255), -1)
        id, con = clf.predict(gray_img[y:y+h, x:x+w])
        print(con)
        if con <= 55:
            cv.putText(img, "View", (x+10, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        else:
            cv.putText(img, "unknown", (x+10, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        show_con = "{0}%".format(round(100 - con))
        cv.rectangle(img, (x+10, y+h+10), (x+w, y+h+50), (255, 0, 255), -1)
        cv.putText(img, show_con, (x+10, y+h+40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        xywh = [x, y, w, h]
    return img, xywh
 
def detect(img, clf):
    img, xywh = draw_boundary(img, clf)
    if len(xywh) == 4:
        result = img[xywh[1]:xywh[1]+xywh[3], xywh[0]:xywh[0]+xywh[2]]
    return img
 
def train_classifier(data_dir):

    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)] 
    faces = []
    ids = []
    for image in path:
        if image == 'data/.DS_Store':
            continue
        if image.endswith(".jpg"):
            img_path = os.path.join(data_dir, image)
        img = Image.open(image).convert("L") 
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image) [1].split("_")[1].split(".")[0])
        faces.append(imageNp) 
        ids.append(id)
    ids = np.array(ids)
    clf = cv.face.LBPHFaceRecognizer_create() 
    clf.train (faces,ids)
    clf.write("classifier.xml")
    print("[INFO] Training completed and saved to classifier.xml")
 
train_classifier("data")
 
clf = cv.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")
 
cap = cv.VideoCapture(0)
img_id = 0
 
while True:
    check, frame = cap.read()
    if not check:
        break
    frame = detect(frame, clf)
    cv.imshow("Output Camera", frame)
    img_id += 1
    if cv.waitKey(1) & 0xFF == ord("g"):
        break
 
cap.release()
cv.destroyAllWindows()
 