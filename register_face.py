import cv2
import csv
import os

STUDENTS_FILE = "students.csv"
FACES_FOLDER = "student_faces"

# Check registered students
if not os.path.exists(STUDENTS_FILE):
    print("Please register a student first!")
    raise SystemExit

with open(STUDENTS_FILE, "r", newline="") as file:
    students = list(csv.DictReader(file))

if not students:
    print("No registered students found!")
    raise SystemExit

print("\nRegistered Students:")
for student in students:
    print(student["Student_ID"], "-", student["Name"])

student_id = input("\nEnter Student ID to capture face: ").strip()

student = next(
    (s for s in students if s["Student_ID"] == student_id),
    None
)

if student is None:
    print("Student ID not found!")
    raise SystemExit

os.makedirs(FACES_FOLDER, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened!")
    raise SystemExit

print("Look at the camera. Press S to save your face image, Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    cv2.imshow("Student Face Registration", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = os.path.join(
            FACES_FOLDER, f"{student_id}.jpg"
        )
        cv2.imwrite(filename, frame)
        print("Face image saved for", student["Name"])
        break

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()