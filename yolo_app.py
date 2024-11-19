import cv2
from ultralytics import YOLO
import time

# Load the model
model = YOLO('new_best_violence_model.pt')

# Define the class names
class_names = {
    0: 'violence',
    1: 'nonviolence'
}

# Open the webcam
cap = cv2.VideoCapture(0)

# Give the camera some time to warm up
time.sleep(2)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame, device='cpu')

    # Draw bounding boxes and labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])  # Get the class ID

            # Get the class name using the dictionary
            class_name = class_names.get(class_id, "Unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_name} {confidence*100:.2f}%'
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


