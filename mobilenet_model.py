import cv2
import time
import datetime
import numpy as np
from tensorflow.keras.models import load_model
# from uploadApi import upload_videos_and_create_objects # used for upload file not cloud

# Load the CNN model
model_path = r'modelnew.h5'  # Replace with your actual path
cnn_model = load_model(model_path)

# Class labels for the CNN model
class_labels = ['Non-Violence', 'Violence']

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open frame")
    exit()

time.sleep(2)

recording = False
start_time = None
incident_active = False
no_violation_duration = 5
out = None

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = 30  # Frames per second (capturing speed)

# Major Changes Made below..................................................
output_fps = input_fps // 6 

# Function to preprocess the frame for the CNN model
def preprocess_frame(frame, target_size=(128, 128)):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Function to detect violence using the CNN model
def detect_violence(frame, model):
    preprocessed_frame = preprocess_frame(frame)
    predictions = model.predict(preprocessed_frame)
    confidence = predictions[0][0]
    detection = confidence > 0.5  # Confidence threshold
    return detection, confidence

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    violence_detected_yolo = False
    cropped_frame = None

    cnn_detection, cnn_confidence = detect_violence(frame, cnn_model)
    if cnn_detection:
        cv2.putText(frame, f'CNN: Violence Detected ({cnn_confidence:.2f})', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'CNN: Non-Violence', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Start recording if either detection is true
    if cnn_detection and not incident_active:
        incident_active = True
        recording = True
        start_time = time.time()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'output_clip_{timestamp}.webm'
        fourcc = cv2.VideoWriter_fourcc(*'vp90')
        out = cv2.VideoWriter(filename, fourcc, output_fps, (frame_width, frame_height))
        print(f"Recording started... Filename: {filename}")

    if recording:
        out.write(frame)
        print("Recording in progress....")
        if time.time() - start_time >= 30:
            recording = False
            out.release()
            upload_videos_and_create_objects(filename)
            print(f"Recording stopped. Saved as {filename}")

    if cnn_detection and incident_active:
        time.sleep(no_violation_duration)
        incident_active = False

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
