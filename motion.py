import cv2 as cv

import numpy as np


cap = cv.VideoCapture(0)
ret, frame1 = cap.read() 
ret, frame2 = cap.read()

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        motiondiff = cv.absdiff(frame1, frame2)
        grayimg = cv.cvtColor(motiondiff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur (grayimg, (5,5),0)
        thresh, result = cv.threshold (blur, 50, 255, cv. THRESH_BINARY)
        dilation = cv.dilate (result, None, iterations=3)
        contours, hierarchy = cv.findContours (dilation, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
        draw_contour = cv.drawContours (frame1, contours, -1, (255,194,216),2)
    for contour in contours :
        (x,y,w,h) = cv.boundingRect(contour) 
        if cv.contourArea (contour) < 2500:
            continue
        cv.rectangle(frame1, (x, y), (x+w, y+h), (200,194,255), 4) 
    cv.imshow('Frame', frame1)
    frame1 = frame2
    ret, frame2 = cap.read() 
    key = cv.waitKey(33)
    if key == ord('q'): 
        break
cap.release() 
cv.destroyAllWindows()


