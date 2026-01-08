import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype='uint8') #blank page
cv.imshow('Blank',blank)

# img = cv.imread("C:/Users/LENOVO/Pictures/Screenshots/UI.png")
# cv.imshow('Test', img)

# blank[200:300,300:400] = 0,0,255 # red
# cv.imshow('Red',blank)

# cv.rectangle(blank,(0,0),(250,500),(0,0,255),thickness=cv.FILLED) #250,500 = width x height
# cv.imshow('Rectangle',blank)
# cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,0,255),thickness=cv.FILLED) #square
# cv.imshow('Rectangle',blank)
# cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40,(0,0,255),thickness=3)
# cv.imshow('Circle',blank)
# cv.line(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,0,255),thickness=3)
# cv.imshow('line',blank)
cv.putText(blank,"Hello",(0,255),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
cv.imshow('Text',blank)
cv.waitKey(0)