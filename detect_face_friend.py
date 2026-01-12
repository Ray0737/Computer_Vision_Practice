import cv2 as cv
from datetime import datetime

# Load separate classifiers
clf_9 = cv.face.LBPHFaceRecognizer_create()
clf_9.read("classifier_9.xml")

clf_alexis = cv.face.LBPHFaceRecognizer_create()
clf_alexis.read("classifier_alexis.xml")

clf_view = cv.face.LBPHFaceRecognizer_create()
clf_view.read("classifier_me.xml")

clf_papang = cv.face.LBPHFaceRecognizer_create()
clf_papang.read("classifier_papang.xml")

# Map classifiers to names
classifiers = {
    "9": clf_9,
    "alexis": clf_alexis,
    "view": clf_view,
    "papang": clf_papang
}

# Threshold for confidence (tune this value)
CONFIDENCE_THRESHOLD = 80

def draw_boundary(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = gray_img[y:y+h, x:x+w]

        # Predict with all classifiers
        predictions = {}
        for name, clf in classifiers.items():
            id, confidence = clf.predict(face_roi)
            predictions[name] = confidence

        # Find classifier with lowest confidence
        best_match = min(predictions, key=predictions.get)
        best_conf = predictions[best_match]

        if best_conf <= CONFIDENCE_THRESHOLD:
            label = best_match
        else:
            label = "unknown"

        cv.putText(img, f"{label} ({round(100-best_conf)}%)", 
                   (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv.putText(img, timestamp, (10, img.shape[0] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return img

# Start webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = draw_boundary(frame)
    cv.imshow("Face Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
