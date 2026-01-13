import cv2
import os

# --- Configuration ---
CLASS_ID = 1         # The '1' in 1.00
START_INDEX = 0       # The '00' in 1.00
TOTAL_IMAGES = 200    # How many images you want to take
SAVE_PATH = "data"    # Folder name

# Create the folder if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
    print(f"Created folder: {SAVE_PATH}")

# Initialize camera
cap = cv2.VideoCapture(0)
count = START_INDEX

print(f"Instructions:")
print(f"1. Press 's' to save a photo")
print(f"2. Press 'q' to quit")

while count < TOTAL_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the live feed
    cv2.imshow("Capture Data - Press 's' to Save", frame)

    key = cv2.waitKey(1)
    
    # Press 's' to save the image
    if key == ord('s'):
        # Formatting filename: {class}.{id:02d} ensures 00, 01, 02 style
        filename = f"{CLASS_ID}.{count:02d}.jpg"
        file_path = os.path.join(SAVE_PATH, filename)
        
        cv2.imwrite(file_path, frame)
        print(f"Saved: {file_path}")
        count += 1
        
    # Press 'q' to quit early
    elif key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"Finished! Captured {count} images.")