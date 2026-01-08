import cv2 as cv
import numpy as np

# Image Show
img = cv.imread("C:/Users/LENOVO/Pictures/Screenshots/UI.png")
cv.imshow('Test', img)
cv.waitKey(0)

#Video Show
capture = cv.VideoCapture('m5python/Aj phoom-my work/work/Sequence 01.mp4')
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video',frame)
    if cv.waitKey(20) & 0xFF ==ord('d'): # if press d
        break
    
capture.release()
cv.destroyAllWindows()

#Resize
img = cv.imread("C:/Users/LENOVO/Pictures/Screenshots/UI.png")

def rescaleFrame(frame,scale=0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def changeRes(width,height):
    capture.set(3,width)
    capture.set(4,height)
    

frame_resized = rescaleFrame(img)
cv.imshow('Resized',frame_resized)
cv.waitKey(0)