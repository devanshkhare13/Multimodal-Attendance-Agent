
import cv2
import csv
import os
from datetime import datetime

FILE_NAME = "attendance.csv"

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")

    # Check whether attendance is already marked today
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, "r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1] == today:
                    print("Attendance already marked today!")
                    return

    file_exists = os.path.exists(FILE_NAME)

    with open(FILE_NAME, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])

        writer.writerow([
            name,
            today,
            datetime.now().strftime("%H:%M:%S")
        ])

    print("Attendance marked for", name)


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened!")
else:
    print("Press A to mark test attendance, Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

        cv2.putText(
            frame, "Faces: " + str(len(faces)),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2
        )

        cv2.imshow("Attendance Agent", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a") and len(faces) > 0:
            mark_attendance("Test Student")

        if key == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
