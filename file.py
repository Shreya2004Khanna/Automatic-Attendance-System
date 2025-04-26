import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image

# Load and encode known faces
def load_encoding(image_path):
    try:
        # Load and convert to RGB
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if too large
        max_dim = max(image.size)
        if max_dim > 800:
            scale = 800 / max_dim
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        image_np = np.array(image)
        encodings = face_recognition.face_encodings(image_np)
        return encodings[0] if encodings else None
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None

# Set path to known faces
photos_path = os.path.join(os.getcwd(), "photos")
known_face_encodings = []
known_face_names = []

# Load known faces
for filename in os.listdir(photos_path):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        name = os.path.splitext(filename)[0]
        path = os.path.join(photos_path, filename)
        encoding = load_encoding(path)
        if encoding is not None:
            known_face_encodings.append(encoding)
            known_face_names.append(name)
        else:
            print(f"Skipping {name}, no face found.")

# Initialize webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open video device.")
    exit()

# Attendance log
attendance = {}

def mark_attendance(name):
    if name not in attendance:
        attendance[name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{name} marked present at {attendance[name]}")

print("Starting webcam recognition. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # speed up
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)

        top, right, bottom, left = [v * 4 for v in face_location]  # scale back up
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (0, 0, 0), 1)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# Save attendance to CSV
with open("attendance.csv", "w", newline="") as f:
    for name, timestamp in attendance.items():
        f.write(f"{name},{timestamp}\n")

print("Attendance saved to attendance.csv")
