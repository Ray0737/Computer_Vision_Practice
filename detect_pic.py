import cv2 as cv
import os
 
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
 
cap = cv.VideoCapture(0)
 
# Set up storage folder
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)
 
img_count = 0  # Counter for saved images
 
while cap.isOpened():
    check, frame = cap.read()
    if check:
        grey_cap = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_dect = face_cascade.detectMultiScale(grey_cap, scaleFactor=1.1, minNeighbors=5)
 
        for (x, y, w, h) in face_dect:
            roi_color = frame[y:y+h, x:x+w]  # Cropped color face
 
            # Save the face image
            img_filename = os.path.join(output_folder, f"face_{img_count}.jpg")
            cv.imwrite(img_filename, roi_color)
            img_count += 1
 
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
 
        cv.imshow('Face Capture', frame)
 
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()