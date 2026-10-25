import cv2
import csv
import os
import sounddevice as sd
import numpy as np
import librosa
from datetime import datetime

# ---------------- SETTINGS ----------------

MODEL_FILE = "face_model.yml"
LABEL_FILE = "face_labels.csv"
ATTENDANCE_FILE = "attendance.csv"

SAMPLE_RATE = 16000
AUDIO_DURATION = 5

# ---------------- CHECK FILES ----------------

if not os.path.exists(MODEL_FILE):
    print("Error: face_model.yml not found. Train the model first.")
    exit()

if not os.path.exists(LABEL_FILE):
    print("Error: face_labels.csv not found.")
    exit()

# ---------------- LOAD FACE MODEL ----------------

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

students = {}

with open(LABEL_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        label = int(row.get("Label", row.get("label", row.get("ID", 0))))

        student_id = row.get(
            "Student_ID",
            row.get("student_id", row.get("StudentID", "Unknown"))
        )

        name = row.get("Name", row.get("name", "Unknown"))

        students[label] = (student_id, name)

if not students:
    print("Error: No students found in face_labels.csv.")
    exit()

print("Face model loaded successfully.")
print("Students registered in model:", len(students))

# ---------------- AUDIO ANALYSIS ----------------

print("\nRecording audio for 5 seconds...")
print("Please speak normally.")

try:
    audio = sd.rec(
        int(AUDIO_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()
    audio = audio.flatten()

    rms_volume = float(np.sqrt(np.mean(audio ** 2)))

    rms_frames = librosa.feature.rms(y=audio)[0]

    threshold = max(
        0.003,
        float(np.percentile(rms_frames, 25)) * 2
    )

    activity_percentage = float(
        np.mean(rms_frames > threshold) * 100
    )

    if rms_volume < 0.003:
        audio_status = "Very Quiet"
    elif activity_percentage > 20:
        audio_status = "High Audio Activity"
    else:
        audio_status = "Low or Moderate Audio Activity"

except Exception as error:
    print("Audio recording failed:", error)
    exit()

# ---------------- ATTENDANCE SETUP ----------------

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])

today = datetime.now().strftime("%Y-%m-%d")

marked_today = set()

with open(ATTENDANCE_FILE, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row.get("Date") == today:
            marked_today.add(row.get("Name", ""))

# Students recognized during the current session
session_present = set()

# ---------------- CAMERA AND FACE RECOGNITION ----------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

print("\nCamera started.")
print("Look at the camera.")
print("Press Q to finish the session.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read from the camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if label in students and confidence < 70:
            student_id, name = students[label]

            session_present.add(name)

            current_time = datetime.now().strftime("%H:%M:%S")

            if name not in marked_today:
                with open(
                    ATTENDANCE_FILE, "a", newline=""
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow([name, today, current_time])

                marked_today.add(name)
                print("Attendance marked for:", name)

            display_text = f"{name} ({student_id})"

            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

            cv2.putText(
                frame, display_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )

        else:
            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 0, 255), 2
            )

            cv2.putText(
                frame, "Unknown", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2
            )

    cv2.imshow("Multimodal Attendance Agent", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

# ---------------- SESSION REPORT ----------------

print("\n===================================")
print("       MULTIMODAL SESSION REPORT")
print("===================================")

print("Session Date:", today)
print("Session Time:", datetime.now().strftime("%H:%M:%S"))

print("\n--- Visual Analysis ---")
print("Students registered in model:", len(students))
print("Students recognized this session:", len(session_present))
print("Students marked present today:", len(marked_today))

if session_present:
    print("Recognized students:", ", ".join(sorted(session_present)))
else:
    print("Recognized students: None")

print("\n--- Audio Analysis ---")
print(f"RMS Volume: {rms_volume:.4f}")
print(f"Audio Activity: {activity_percentage:.2f}%")
print("Audio Status:", audio_status)

# ---------------- ENGAGEMENT SCORING ----------------

presence_score = 100 if len(session_present) > 0 else 0

audio_score = min(activity_percentage * 2, 100)

engagement_score = (
    0.5 * presence_score +
    0.5 * audio_score
)

print("\n--- Combined Result ---")

if session_present and activity_percentage > 20:
    engagement_status = "Presence detected with high audio activity"
elif session_present:
    engagement_status = "Presence detected with low or moderate audio activity"
else:
    engagement_status = "No recognized student detected in this session"

print("Session Indicator:", engagement_status)
print(f"Presence Score: {presence_score:.2f}/100")
print(f"Audio Activity Score: {audio_score:.2f}/100")
print(f"Experimental Engagement Score: {engagement_score:.2f}/100")

print("\nNote: Audio activity does not prove student attention.")
print("This is an experimental indicator, not a validated engagement score.")

# ---------------- EPIC 5.1: TEACHING SUGGESTIONS ----------------

print("\n--- Teaching Suggestions ---")

if len(session_present) == 0:
    print("Suggestion: Check student attendance and camera positioning.")

elif rms_volume < 0.003:
    print("Suggestion: Check whether the microphone is working.")
    print("Suggestion: Verify that classroom audio is being captured.")

elif activity_percentage > 20:
    print("Suggestion: Review the audio context before deciding whether to clarify the topic.")
    print("Note: High audio activity does not necessarily mean high engagement.")

else:
    print("Suggestion: Check audio clarity and encourage student participation.")

print("Teaching suggestions generated successfully!")
# ---------------- EPIC 5.2: SESSION ANALYTICS ----------------

import csv
import os

analytics_file = "session_analytics.csv"

file_exists = os.path.isfile(analytics_file)

with open(analytics_file, "a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "Session_Date",
            "Session_Time",
            "Students_Recognized",
            "Recognized_Names",
            "RMS_Volume",
            "Audio_Activity_Percentage",
            "Audio_Status",
            "Engagement_Score"
        ])

    writer.writerow([
        datetime.now().strftime("%Y-%m-%d"),
        datetime.now().strftime("%H:%M:%S"),
        len(session_present),
        ", ".join(sorted(session_present)),
        round(rms_volume, 4),
        round(activity_percentage, 2),
        audio_status,
        round(engagement_score, 2)
    ])

print("Session analytics saved to", analytics_file)

print("\nMultimodal session report completed!")
