import cv2
import os
import csv
import numpy as np

FACE_FOLDER = "student_faces"
STUDENT_FILE = "students.csv"

faces = []
labels = []
student_details = {}

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Read student information
with open(STUDENT_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for label, row in enumerate(reader):
        student_id = row["Student_ID"].strip()
        name = row["Name"].strip()

        image_path = os.path.join(FACE_FOLDER, student_id + ".jpg")

        if not os.path.exists(image_path):
            print("Image not found:", image_path)
            continue

        image = cv2.imread(image_path)

        if image is None:
            print("Could not read image:", image_path)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(detected_faces) == 0:
            print("No face detected in:", image_path)
            continue

        # Select the largest detected face
        x, y, w, h = max(
            detected_faces,
            key=lambda face: face[2] * face[3]
        )

        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))

        faces.append(face)
        labels.append(label)
        student_details[label] = (student_id, name)

if len(faces) == 0:
    print("No usable face images found.")
    print("Check your student_faces folder and registered images.")
    raise SystemExit

# Train the face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels, dtype=np.int32))
recognizer.save("face_model.yml")

# Save student ID and name mapping
with open("face_labels.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Label", "Student_ID", "Name"])

    for label, (student_id, name) in student_details.items():
        writer.writerow([label, student_id, name])

print("\nModel training completed!")
print("Saved model: face_model.yml")
print("Registered students:", len(student_details))