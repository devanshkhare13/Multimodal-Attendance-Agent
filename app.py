from flask import Flask, Response, render_template_string
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Multimodal Attendance Agent</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background: #eef2f7;
            padding: 30px;
        }
        h1 { color: #243b55; }
        img {
            width: 640px;
            max-width: 95%;
            border: 4px solid #243b55;
            border-radius: 12px;
        }
        p { color: #444; }
    </style>
</head>
<body>
    <h1>Multimodal Attendance Agent</h1>
    <h3>Student Face Registration</h3>
    <img src="/video_feed" alt="Camera feed">
    <p>Camera feed is running in your browser.</p>
</body>
</html>
"""

def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    if not camera.isOpened():
        print("Camera could not be opened!")
    else:
        print("Starting Attendance Agent...")
        print("Open http://127.0.0.1:5000 in your browser")

    app.run(debug=False, use_reloader=False)