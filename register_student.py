import csv
import os

FILE_NAME = "students.csv"

def register_student():
    student_id = input("Enter Student ID: ").strip()
    name = input("Enter Student Name: ").strip()

    if not student_id or not name:
        print("Student ID and name cannot be empty!")
        return

    students = []

    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, "r", newline="") as file:
            students = list(csv.reader(file))

    for student in students[1:]:
        if student and student[0] == student_id:
            print("This Student ID already exists!")
            return

    file_exists = os.path.exists(FILE_NAME)

    with open(FILE_NAME, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists or os.path.getsize(FILE_NAME) == 0:
            writer.writerow(["Student_ID", "Name"])

        writer.writerow([student_id, name])

    print("Student registered successfully!")

register_student()