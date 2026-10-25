import cv2
import csv
import os
from datetime import datetime

MODEL_FILE = "face_model.yml"
LABEL_FILE = "face_labels.csv"
ATTENDANCE_FILE = "attendance.csv"

# Check required files
if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_FILE):
    print("Model files missing. Run train_model.py first.")
    raise SystemExit

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

# Load student details
students = {}

with open(LABEL_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        label = int(row["Label"])
        students[label] = (
            row["Student_ID"],
            row["Name"]
        )

# Create attendance file if it doesn't exist
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])

# Read students already marked present today
today = datetime.now().strftime("%Y-%m-%d")
marked_today = set()

with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row.get("Date") == today:
            marked_today.add(row.get("Name", "").strip())

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened!")
    raise SystemExit

print("Automatic attendance started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if label in students and confidence < 70:
            student_id, name = students[label]

            if name not in marked_today:
                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                with open(
                    ATTENDANCE_FILE, "a",
                    newline="", encoding="utf-8"
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow([name, date, time])

                marked_today.add(name)
                print(f"Attendance marked: {name} ({student_id})")

            display_text = f"{name} - Present"

        else:
            display_text = "Unknown Student"

        cv2.rectangle(
            frame, (x, y), (x + w, y + h), (0, 255, 0), 2
        )

        cv2.putText(
            frame, display_text,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 0), 2
        )

    cv2.imshow("Automatic Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()