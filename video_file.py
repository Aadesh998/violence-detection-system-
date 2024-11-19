import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load the YOLO model
yolo_model = YOLO('best_violence_model.pt')

# Load the CNN model
model_path = r'modelnew.h5'  # Replace with your actual path
cnn_model = load_model(model_path)

# Class labels for the CNN model
class_labels = ['Non-Violence', 'Violence']

def preprocess_frame(frame, target_size=(128, 128)):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def detect_violence(frame, model):
    preprocessed_frame = preprocess_frame(frame)
    predictions = model.predict(preprocessed_frame)
    confidence = predictions[0][0]
    detection = confidence > 0.5  # Confidence threshold
    return detection, confidence

def predict_and_plot_video(video_file, output_path):
    """
    Process a video file for violence detection, annotate detected frames,
    and save the output video.
    
    Args:
        video_file (str): Path to the input video file.
        output_path (str): Path to save the output annotated video.

    Returns:
        str: Path to the saved output video.
    """
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'vp90')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
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
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped_frame = frame[y1:y2, x1:x2]  # Crop the detected area
                        break  # Stop checking if a relevant class is found

                if violence_detected_yolo:
                    break  # Exit the outer loop if violence is detected

            # If YOLO detected something, classify the cropped frame using CNN
            if violence_detected_yolo and cropped_frame is not None:
                cnn_detection, cnn_confidence = detect_violence(cropped_frame, cnn_model)
                cv2.putText(frame, f'CNN: Violence Detected ({cnn_confidence:.2f})', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                # Pass the complete frame to the CNN model if nothing is detected by YOLO
                cnn_detection, cnn_confidence = detect_violence(frame, cnn_model)
                if cnn_detection:
                    cv2.putText(frame, f'CNN: Violence Detected ({cnn_confidence:.2f})', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'CNN: Non-Violence', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Write the annotated frame to the output video
            out.write(frame)

        cap.release()
        out.release()
        return output_path
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return None


predict_and_plot_video(r'crop.mp4', r'output_file.webm')