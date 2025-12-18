import cv2
import mediapipe as mp
import numpy as np
import json
from tensorflow.keras.models import load_model
from collections import deque

class ASLRecognizer:
    def __init__(self, model_path='models/best_model.keras', label_map_path='processed_data/label_map.json'):
        print("Initializing ASL Recognizer...")
        
        try:
            self.model = load_model(model_path)
            print(f" Model loaded from {model_path}")
        except:
            print(f"Error loading model from {model_path}")
            exit(1)
        
        try:
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            self.label_map = {int(k): v for k, v in self.label_map.items()}
            print(f"Loaded {len(self.label_map)} labels")
        except:
            print(f"Error loading label map from {label_map_path}")
            exit(1)
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.prediction_buffer = deque(maxlen=5)
        
        print("MediaPipe initialized")
        print("\nReady! Press 'q' to quit\n")
    
    def extract_landmarks(self, hand_landmarks):
        coords = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        coords = np.array(coords).flatten()
        
        wrist = coords[:3]
        coords = coords - np.tile(wrist, 21)
        
        max_val = np.max(np.abs(coords))
        if max_val > 0:
            coords = coords / max_val
        
        return coords.reshape(1, 21, 3)
    
    def get_smoothed_prediction(self, prediction):
        self.prediction_buffer.append(prediction)
        
        if len(self.prediction_buffer) >= 3:
            avg_prediction = np.mean(list(self.prediction_buffer), axis=0)
            return avg_prediction
        return prediction
    
    def draw_info_panel(self, frame, letter, confidence, fps):
        height, width = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        cv2.putText(frame, f"Letter: {letter}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        cv2.putText(frame, "Confidence:", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        bar_width = int(250 * confidence)
        cv2.rectangle(frame, (20, 100), (270, 120), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 100), (20 + bar_width, 120), color, -1)
        cv2.putText(frame, f"{confidence:.2%}", (280, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        fps_counter = deque(maxlen=30)
        import time
        
        print("Camera opened successfully!")
        print("Instructions:")
        print("- Show ASL hand signs to the camera")
        print("- Keep your hand in frame and well-lit")
        print("- Press 'q' to quit")
        
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret:
                print("Error reading frame")
                break
            
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            letter = "No hand detected"
            confidence = 0.0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    landmarks = self.extract_landmarks(hand_landmarks)
                    
                    prediction = self.model.predict(landmarks, verbose=0)[0]
                    prediction = self.get_smoothed_prediction(prediction)
                    
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[predicted_class]
                    
                    if confidence > 0.5:
                        letter = self.label_map.get(predicted_class, "Unknown")
            
            fps = 1.0 / (time.time() - start_time)
            fps_counter.append(fps)
            avg_fps = np.mean(fps_counter)
            
            frame = self.draw_info_panel(frame, letter, confidence, avg_fps)
            
            cv2.putText(frame, "Press 'q' to quit", (width - 200, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('ASL Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\nCamera closed. Goodbye!")

if __name__ == "__main__":
    recognizer = ASLRecognizer()
    recognizer.run()