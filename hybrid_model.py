import cv2
from ultralytics import YOLO
import time
import datetime
import numpy as np
from tensorflow.keras.models import load_model

# Load the YOLO model
yolo_model = YOLO('new_best_violence_model.pt')

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
no_violation_duration = 15
out = None

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = 30  # Frames per second (capturing speed)

# Major Changes Made below..................................................
output_fps = input_fps // 6 
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# fourcc = cv2.VideoWriter_fourcc(*'vp90')


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
    results = yolo_model.predict(rgb_frame, device='cpu')

    violence_detected_yolo = False
    cropped_frame = None

    # Check YOLO results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            # Check if YOLO detects any relevant class
            if class_id in [0, 1, 2, 3, 4, 6, 7]:  
                violence_detected_yolo = True
                # Crop the detected region
                # x1, y1, x2, y2 = map(int, box.xyxy[0])
                # cropped_frame = frame[y1:y2, x1:x2]  # Crop the detected area
                break  # Stop checking if a relevant class is found

        if violence_detected_yolo:
            break  # Exit the outer loop if violence is detected

    # If YOLO detected something, classify the cropped frame using CNN
    # if violence_detected_yolo and cropped_frame is not None:
    #     cnn_detection, cnn_confidence = detect_violence(cropped_frame, cnn_model)
    #     cv2.putText(frame, f'CNN: Violence Detected ({cnn_confidence:.2f})', (10, 30), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # else:
        # Pass the complete frame to the CNN model if nothing is detected by YOLO
        cnn_detection, cnn_confidence = detect_violence(frame, cnn_model)
        if cnn_detection:
            cv2.putText(frame, f'CNN: Violence Detected ({cnn_confidence:.2f})', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'CNN: Non-Violence', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Start recording if either detection is true
    if (violence_detected_yolo or cnn_detection) and not incident_active:
        incident_active = True
        recording = True
        start_time = time.time()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'video/output_clip_{timestamp}.webm'
        fourcc = cv2.VideoWriter_fourcc(*'vp90')
        out = cv2.VideoWriter(filename, fourcc, output_fps, (frame_width, frame_height))
        print(f"Recording started... Filename: {filename}")

    if recording:
        out.write(frame)
        print("Recording in progress....")
        if time.time() - start_time >= 30:
            recording = False
            out.release()
            print(f"Recording stopped. Saved as {filename}")

    if not (violence_detected_yolo or cnn_detection) and incident_active:
        time.sleep(no_violation_duration)
        incident_active = False

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
