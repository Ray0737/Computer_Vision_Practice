import cv2 as cv
import numpy as np

#--------------------------------------------------Image Show--------------------------------------------------#
img = cv.imread("m5python/Aj phoom-my work/023546_school/R.jpg")
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

#--------------------------------------------------Resize--------------------------------------------------#
img = cv.imread("m5python/Aj phoom-my work/023546_school/R.jpg")

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

#--------------------------------------------------Blank Page--------------------------------------------------#

blank = np.zeros((500,500,3),dtype='uint8') #blank page
cv.imshow('Blank',blank)

img = cv.imread("m5python/Aj phoom-my work/023546_school/R.jpg")
cv.imshow('Test', img)

blank[200:300,300:400] = 0,0,255 # red
cv.imshow('Red',blank)

cv.rectangle(blank,(0,0),(250,500),(0,0,255),thickness=cv.FILLED) #250,500 = width x height
cv.imshow('Rectangle',blank)
cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,0,255),thickness=cv.FILLED) #square
cv.imshow('Rectangle',blank)
cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40,(0,0,255),thickness=3)
cv.imshow('Circle',blank)
cv.line(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,0,255),thickness=3)
cv.imshow('line',blank)
cv.putText(blank,"Hello",(0,255),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
cv.imshow('Text',blank)
cv.waitKey(0)

#--------------------------------------------------Crop--------------------------------------------------#

img = cv.imread('m5python/Aj phoom-my work/023546_school/R.jpg')
# Select rows 80 to 280 and columns 150 to 330
cropped_image = img[80:280, 150:330] 

cv.imshow("cropped", cropped_image)
cv.imwrite("Cropped Image.jpg", cropped_image) # Saves the new image
cv.waitKey(0)

#--------------------------------------------------Rotation--------------------------------------------------#

# Load the image
img = cv.imread('m5python/Aj phoom-my work/023546_school/R.jpg')
(h, w) = img.shape[:2] # Get image height and width

# 1. Define the rotation parameters
center = (w // 2, h // 2)  # Rotate around the center of the image
angle = 45                 # Degrees to rotate (positive for counter-clockwise)
scale = 1.0                # Keep the same scale (no zoom)

# 2. Get the rotation matrix (OpenCV helper function)
# This creates the 2x3 matrix mentioned in the document
matrix = cv.getRotationMatrix2D(center, angle, scale)

# 3. Apply the rotation using warpAffine
# This moves every pixel to its new rotated position
rotated_image = cv.warpAffine(img, matrix, (w, h))

# Display the results
cv.imshow("Original", img)
cv.imshow("Rotated Image", rotated_image)
cv.waitKey(0)
cv.destroyAllWindows()

# --------------------------------------------------Mouse Click Show text--------------------------------------------------#

img = cv.imread('R.jpg')

def click_position(event, x, y, flags, param):
    # Check if the event is a left mouse button click
    if event == cv.EVENT_LBUTTONDOWN:
        # Draw the text 'totoro' on the image at the clicked coordinates (x, y)
        # 1: Font type (FONT_HERSHEY_SIMPLEX)
        # 3: Font scale (size)
        # (255, 223, 194): Color in BGR format (Light blue/Lavender)
        # 3: Thickness of the text lines
        cv.putText(img, 'totoro', (x, y), 1, 3, (255, 223, 194), 3)
        
        cv.imshow('click puttext', img)


cv.imshow('click puttext', img)

# Link the mouse callback function to the specific window named 'click puttext'
cv.setMouseCallback('click puttext', click_position)


cv.waitKey(0)
cv.destroyAllWindows()

