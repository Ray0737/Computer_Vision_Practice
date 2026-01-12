import cv2 as cv
import numpy as np

vid = cv.VideoCapture('way.mp4')
scale = 0.3

while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv.resize(frame, (0, 0), fx=scale, fy=scale)    
    roi = frame

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9,9), 0)
    edges = cv.Canny(blurred, 140, 320, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('Original', frame)
    cv.imshow('ROI with Lines', roi)
    cv.imshow('Edges (Canny)', edges)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
