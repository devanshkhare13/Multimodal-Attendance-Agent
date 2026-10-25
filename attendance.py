
import csv
import os
from datetime import datetime

FILE_NAME = "attendance.csv"

def mark_attendance(student_name):
    file_exists = os.path.exists(FILE_NAME)

    with open(FILE_NAME, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])

        now = datetime.now()
        writer.writerow([
            student_name,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S")
        ])

    print("Attendance marked for", student_name)

mark_attendance("Student 1")
