import cv2 as cv
import os
import time

# --- Setup ---
if not os.path.exists("captured_faces"):
    os.makedirs("captured_faces")

cap = cv.VideoCapture(0)
xml_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv.CascadeClassifier(xml_path)
clf = cv.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

names = {
    1: "Gain"
}

# --- Motion detection setup ---
ret, frame1 = cap.read()
ret, frame2 = cap.read()

filled_ids = set()

# --- Countdown setup ---
countdown_secs = 10
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    elapsed = int(time.time() - start_time)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_detect = face_cascade.detectMultiScale(gray_img, 1.1, 5)

    if elapsed < countdown_secs:
        # Show countdown on frame
        remaining = countdown_secs - elapsed
        cv.putText(frame, f"Countdown: {remaining}s", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        # Draw green boxes and labels only
        for (x, y, w, h) in face_detect:
            face_img = gray_img[y:y + h, x:x + w]
            id, con = clf.predict(face_img)
            name = names.get(id, "unknown")
            label = f"{name} ({id})"
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.imshow("Face & Motion Detection", frame)
        frame1 = frame2
        ret, frame2 = cap.read()
        if cv.waitKey(33) & 0xFF == ord('q'):
            break
        continue

    # --- Motion detection ---
    motiondiff = cv.absdiff(frame1, frame2)
    gray_motion = cv.cvtColor(motiondiff, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray_motion, (5, 5), 0)
    _, result = cv.threshold(blur, 50, 255, cv.THRESH_BINARY)
    dilation = cv.dilate(result, None, iterations=3)
    contours, _ = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for (x, y, w, h) in face_detect:
        face_img = gray_img[y:y + h, x:x + w]
        id, con = clf.predict(face_img)
        name = names.get(id, "unknown")
        label = f"{name} ({id})"

        # Check for movement in the face region
        moving = False
        for contour in contours:
            cx, cy, cw, ch = cv.boundingRect(contour)
            if cv.contourArea(contour) < 2500:
                continue
            if (x < cx + cw and x + w > cx and y < cy + ch and y + h > cy):
                moving = True
                break

        # Only add this id if this face moved
        if moving:
            filled_ids.add(id)

        # Only fill the bounding box for this id if it has moved
        if id in filled_ids:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), -1)
        else:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Find all unique IDs in the current frame
    all_ids = set()
    for (x, y, w, h) in face_detect:
        face_img = gray_img[y:y + h, x:x + w]
        id, con = clf.predict(face_img)
        all_ids.add(id)

    # If all but one face have filled boxes, declare the last as winner
    if len(all_ids) - len(filled_ids) == 1 and len(filled_ids) >= 1:
        winner_ids = all_ids - filled_ids
        if winner_ids:
            winner_id = winner_ids.pop()
            winner_name = names.get(winner_id, "unknown")
            win_text = f"{winner_name} wins the game!"
            cv.putText(frame, win_text, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    cv.imshow("Face & Motion Detection", frame)
    frame1 = frame2
    ret, frame2 = cap.read()
    if cv.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()