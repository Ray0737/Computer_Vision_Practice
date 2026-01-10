import cv2
import os
import time

# --- Setup ---
# Check if the folder for saving images exists; if not, create it
if not os.path.exists("captured_faces"):
    os.makedirs("captured_faces")

# --- Setup face recog ---
# Initialize the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade for detecting front-facing faces
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# --- Setup Model ---
# Create an LBPH (Local Binary Patterns Histograms) Face Recognizer
clf = cv2.face.LBPHFaceRecognizer_create()
# Load your custom trained model (which contains data for the people listed below)
clf.read("four.xml")

# Mapping of ID numbers to names
names = {
    1: "mark",
    2: "august",
    3: "the black one",
    4: "zen"
}

# --- Motion detection setup ---
# Capture two initial frames to compare for the difference (motion)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# A set to keep track of IDs who have been caught moving
filled_ids = set()

# --- Countdown setup ---
countdown_secs = 10
start_time = time.time() # Get the current time to start the clock

while True:
    # Read current frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate how many seconds have passed since the script started
    elapsed = int(time.time() - start_time) 
    
    # Convert frame to grayscale (required for face detection and recognition)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    face_detect = face_cascade.detectMultiScale(gray_img, 1.1, 5) 

    # --- PHASE 1: Countdown (Pre-game) ---
    if elapsed < countdown_secs:
        remaining = countdown_secs - elapsed
        # Draw the countdown timer on the screen
        cv2.putText(frame, f"Countdown: {remaining}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        
        # Identify faces but don't check for motion yet
        for (x, y, w, h) in face_detect:
            face_img = gray_img[y:y + h, x:x + w]
            id, con = clf.predict(face_img) # Predict who the person is
            name = names.get(id, "unknown")
            label = f"{name} ({id})"
            # Draw green rectangle for "safe" period
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Face & Motion Detection", frame)
        
        # Update frames for motion detection buffer
        frame1 = frame2
        ret, frame2 = cap.read()
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
        continue # Skip the motion detection logic below until countdown is over

    # --- PHASE 2: Motion detection (The Game) ---
    # Calculate the absolute difference between two consecutive frames
    motiondiff = cv2.absdiff(frame1, frame2)
    gray_motion = cv2.cvtColor(motiondiff, cv2.COLOR_BGR2GRAY)
    
    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(gray_motion, (5, 5), 0)
    
    # Threshold the image: if a pixel changed significantly, it becomes white (255)
    _, result = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    
    # Dilate the white spots to make movement areas more solid
    dilation = cv2.dilate(result, None, iterations=3)
    
    # Find contours (outlines) of the moving areas
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for (x, y, w, h) in face_detect:
        face_img = gray_img[y:y + h, x:x + w]
        id, con = clf.predict(face_img)
        name = names.get(id, "unknown") 
        label = f"{name} ({id})"

        # Check if any moving contour overlaps with the face's bounding box
        moving = False
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            # Ignore very small movements (noise)
            if cv2.contourArea(contour) < 2500:
                continue
            # Check for collision (overlap) between face box and motion box
            if (x < cx + cw and x + w > cx and y < cy + ch and y + h > cy):
                moving = True
                break

        # If movement was detected in the face area, add to "eliminated" list
        if moving:
            filled_ids.add(id)

        # If person has moved, fill their box with solid red; otherwise, keep it green
        if id in filled_ids:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), -1) # -1 fills the box
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- WINNER LOGIC ---
    # Get all unique face IDs currently visible in the frame
    all_ids = set()
    for (x, y, w, h) in face_detect:
        face_img = gray_img[y:y + h, x:x + w]
        id, con = clf.predict(face_img)
        all_ids.add(id)

    # If only one person is left who hasn't moved (and at least one person HAS moved)
    if len(all_ids) - len(filled_ids) == 1 and len(filled_ids) >= 1:
        winner_ids = all_ids - filled_ids # Find the ID not in the "filled" set
        if winner_ids:
            winner_id = winner_ids.pop()
            winner_name = names.get(winner_id, "unknown")
            win_text = f"{winner_name} wins the game!"
            cv2.putText(frame, win_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # Show the final processed frame
    cv2.imshow("Face & Motion Detection", frame)
    
    # Update frame buffer for next loop iteration
    frame1 = frame2
    ret, frame2 = cap.read()
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
