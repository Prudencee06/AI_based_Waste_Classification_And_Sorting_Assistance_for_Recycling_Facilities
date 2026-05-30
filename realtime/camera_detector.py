# Real-time waste detection using YOLO for object detection + Custom waste classifier.

import cv2
import numpy as np
import requests
import time
import sys
import os
import signal
import winsound

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.audio_alert import AudioAlert
from ml_model.waste_classifier import WasteClassifier

from ultralytics import YOLO as UltralyticsYOLO


class RealTimeDetector:
    def __init__(self, api_base_url="http://127.0.0.1:8000/api"):
        self.api_base_url = api_base_url
        
        # Loads custom waste classifier (trained on plastic, metal, paper, glass, cardboard)
        print("Loading custom waste classifier...")
        try:
            self.waste_classifier = WasteClassifier()
            print(" Custom waste classifier loaded!")
        except Exception as e:
            print(f" Error loading waste classifier: {e}")
            self.waste_classifier = None
        
        # Load YOLO for object detection
        print("Loading YOLO for object detection...")
        try:
            self.yolo_model = UltralyticsYOLO('yolo12n.pt')
            print(" YOLO model loaded!")
        except Exception as e:
            print(f" Error loading YOLO: {e}")
            self.yolo_model = None
        
        self.alert = AudioAlert()
        
        # Session tracking
        self.session_id = None
        self.session_active = False
        self.running = True
        
        # Configuration
        self.confidence_threshold = 0.3
        self.audio_enabled = True
        self.target_stream = None  # 'plastic', 'paper', 'metal', 'glass'
        
        # Statistics
        self.detection_count = 0
        self.alert_count = 0
        
        # Colors for waste types (BGR)
        self.colors = {
            'plastic': (255, 0, 0),      # Blue
            'metal': (0, 0, 255),        # Red
            'paper': (0, 255, 0),        # Green
            'glass': (255, 255, 0),      # Cyan
            'cardboard': (0, 165, 255),  # Orange
            'default': (255, 255, 255)   # White
        }
        
        # YOLO classes that might contain waste
        self.relevant_yolo_classes = [
            'bottle', 'cup', 'can', 'knife', 'fork', 'spoon', 
            'scissors', 'bowl', 'plate', 'jar', 'glass', 'plastic'
        ]
        
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print("\n\nShutting down...")
        self.running = False
    
    def classify_with_custom_model(self, roi):

        # Useing custom trained model to classify waste (plastic, metal, paper, glass, cardboard).

        if roi is None or roi.size == 0:
            return None, 0.0
        
        if self.waste_classifier is None:
            return None, 0.0
        
        try:
            roi_resized = cv2.resize(roi, (224, 224))
            result = self.waste_classifier.classify_image(roi_resized)
            
            if 'error' not in result:
                return result['label'], result['confidence']
            else:
                return None, 0.0
        except Exception as e:
            return None, 0.0
    
    def play_beep(self):
        # Play a beep sound for contamination alerts.
        try:
            winsound.Beep(1000, 500)
        except:
            try:
                print('\a', end='', flush=True)
            except:
                pass
    
    def draw_bounding_box(self, frame, box, waste_type, confidence, is_contaminant, yolo_class=None):
        # Draw bounding box with waste type label.
        x1, y1, x2, y2 = map(int, box)
        color = self.colors.get(waste_type, self.colors['default'])
        
        if is_contaminant:
            color = (0, 0, 255)  # Red for contaminants
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Prepare label
        alert_symbol = "⚠️ " if is_contaminant else ""
        label_text = f"{alert_symbol}{waste_type.upper()} {confidence:.1%}"
        
        # Draw label background
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_x = x1 + (x2 - x1) // 2 - text_w // 2
        label_y = y2 + text_h + 10
        
        if label_y + text_h > frame.shape[0]:
            label_y = y1 - 10
        
        cv2.rectangle(frame, (label_x - 5, label_y - text_h - 5),
                      (label_x + text_w + 5, label_y + 5), color, -1)
        cv2.putText(frame, label_text, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_hud(self, frame, fps):
        """Draw heads-up display."""
        h, w = frame.shape[:2]
        
        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Session info
        session_str = str(self.session_id) if self.session_id else "None"
        target_str = self.target_stream.upper() if self.target_stream else "NOT SET"
        
        cv2.putText(frame, f"Session: {session_str}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Target Stream: {target_str}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Instructions
        cv2.putText(frame, "Press 's' to set target | 'a' for audio | 'q' to quit", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Audio indicator
        audio_status = "🔊" if self.audio_enabled else "🔇"
        cv2.putText(frame, audio_status, (w - 40, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if self.audio_enabled else (100, 100, 100), 1)
        
        return frame
    
    def draw_controls(self, frame):
        """Draw control hints at bottom."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 35), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        controls = "[ESC/Q] Quit  [A] Audio  [S] Set Target  [C] Stats"
        cv2.putText(frame, controls, (w // 2 - 220, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame
    
    def check_contamination(self, waste_type):
        # Check if detected waste type is a contaminant for current target stream.
        if not self.target_stream:
            return False
        
        # METAL is ALWAYS a contaminant unless sorting metal
        if waste_type == 'metal' and self.target_stream != 'metal':
            return True
        
        contamination_rules = {
            'plastic': ['metal', 'glass'],
            'paper': ['metal', 'glass'],
            'cardboard': ['metal', 'glass'],
            'glass': ['metal'],
            'metal': []
        }
        
        return waste_type in contamination_rules.get(self.target_stream, [])
    
    def start_session(self, operator_name=None, shift=None):
        try:
            if not shift:
                hour = time.localtime().tm_hour
                if 6 <= hour < 14:
                    shift = 'morning'
                elif 14 <= hour < 22:
                    shift = 'afternoon'
                else:
                    shift = 'night'
            
            if not operator_name:
                operator_name = 'AI Sorting Assistant'
            
            response = requests.post(
                f"{self.api_base_url}/sessions/",
                json={'operator_name': operator_name, 'shift': shift},
                timeout=5
            )
            
            if response.status_code == 201:
                session_data = response.json()
                self.session_id = session_data['id']
                self.session_active = True
                print(f"\n Session started! ID: {self.session_id}")
                return True
            return False
        except Exception as e:
            print(f" Error starting session: {e}")
            return False
    
    def end_session(self):
        if not self.session_active or not self.session_id:
            return
        
        try:
            response = requests.post(
                f"{self.api_base_url}/sessions/{self.session_id}/end/",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n Session Summary:")
                print(f"   Total Detections: {data['total_detections']}")
                print(f"   Total Alerts: {data['total_alerts']}")
                if data.get('average_confidence'):
                    print(f"   Avg Confidence: {data['average_confidence']:.1%}")
                print(f"\n Session ended!")
        except Exception as e:
            print(f" Error ending session: {e}")
        
        self.session_active = False
        self.session_id = None
    
    def log_detection(self, waste_type, confidence, alert_triggered=False, alert_reason=None):
        if not self.session_active:
            return
        try:
            data = {
                'detected_class': waste_type,
                'confidence_score': confidence,
                'alert_triggered': alert_triggered,
                'alert_reason': alert_reason,
                'session_id': str(self.session_id),
                'event_type': 'alert' if alert_triggered else 'detection'
            }
            requests.post(f"{self.api_base_url}/detections/", json=data, timeout=3)
        except:
            pass
    
    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print(" Error: Could not open camera")
            return
        
        print("\n" + "=" * 60)
        print("   WASTE CLASSIFICATION SYSTEM")
        print("   For Conveyor Belt Sorting Assistance")
        print("=" * 60)
        
        operator = input("Operator name (Enter for default): ").strip()
        shift = input("Shift (morning/afternoon/night, Enter for auto): ").strip()
        
        if not self.start_session(operator if operator else None, shift if shift else None):
            cap.release()
            return
        
        print("\n" + "=" * 60)
        print("   HOW TO USE")
        print("=" * 60)
        print("1. Press 's' to set what you're sorting (plastic/paper/metal/glass)")
        print("2. Place waste items in front of camera")
        print("3. System will classify and show bounding boxes")
        print("4. If contamination (e.g., metal in plastic) is detected, you'll hear a BEEP!")
        print("=" * 60)
        print("\n Starting camera...\n")
        
        frame_count = 0
        start_time = time.time()
        last_alert_time = {}
        last_detection_time = {}
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Use YOLO to find objects
            if self.yolo_model:
                results = self.yolo_model(frame, verbose=False)
                
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            confidence = float(box.conf[0])
                            
                            if confidence < self.confidence_threshold:
                                continue
                            
                            class_id = int(box.cls[0])
                            yolo_class = self.yolo_model.names[class_id]
                            
                            # Extract region for custom classification
                            roi = frame[y1:y2, x1:x2]
                            
                            # Use YOUR custom waste classifier!
                            waste_type, waste_conf = self.classify_with_custom_model(roi)
                            
                            if waste_type and waste_conf > 0.3:
                                # Print detection
                                print(f"   {yolo_class} -> {waste_type.upper()} ({waste_conf:.1%})")
                                
                                # Check if this is contamination
                                is_contaminant = self.check_contamination(waste_type)
                                
                                # Draw bounding box
                                frame = self.draw_bounding_box(
                                    frame, (x1, y1, x2, y2),
                                    waste_type, waste_conf,
                                    is_contaminant, yolo_class
                                )
                                
                                # Handle alert for contamination
                                if is_contaminant and waste_conf > 0.35:
                                    key = f"{waste_type}_{int(x1/10)}_{int(y1/10)}"
                                    current_time = time.time()
                                    if key not in last_alert_time or current_time - last_alert_time[key] > 2:
                                        last_alert_time[key] = current_time
                                        self.alert_count += 1
                                        self.detection_count += 1
                                        
                                        if self.audio_enabled:
                                            self.play_beep()
                                            print(f"\n CONTAMINATION ALERT!!!!!!! ")
                                            print(f"   Detected: {waste_type.upper()} ({waste_conf:.1%})")
                                            target_name = self.target_stream.upper() if self.target_stream else "UNKNOWN"
                                            print(f"   Target Stream: {target_name}")
                                            print(f"   ACTION REQUIRED: Remove from line!\n")
                                        
                                        self.log_detection(
                                            waste_type, waste_conf,
                                            alert_triggered=True,
                                            alert_reason=f"{waste_type} in {self.target_stream} stream"
                                        )
                                else:
                                    # Log normal detection periodically
                                    item_key = f"{waste_type}"
                                    current_time = time.time()
                                    if item_key not in last_detection_time or current_time - last_detection_time[item_key] > 3:
                                        last_detection_time[item_key] = current_time
                                        self.detection_count += 1
                                        self.log_detection(waste_type, waste_conf, alert_triggered=False)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Draw HUD
            frame = self.draw_hud(frame, fps)
            frame = self.draw_controls(frame)
            
            cv2.imshow('Waste Classification System - Conveyor Belt Assistant', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                self.running = False
            elif key == ord('a'):
                self.audio_enabled = not self.audio_enabled
                status = "ON " if self.audio_enabled else "OFF 🔇"
                print(f"\n Audio alerts: {status}")
                if self.audio_enabled:
                    self.play_beep()
                    print("   Test beep played!")
            elif key == ord('s'):
                print("\n" + "-" * 40)
                print("SET TARGET STREAM (What are you sorting today?)")
                print("-" * 40)
                print("Options: plastic, paper, metal, glass")
                stream = input("Enter target stream: ").lower()
                if stream in ['plastic', 'paper', 'metal', 'glass']:
                    self.target_stream = stream
                    print(f"\n Target stream set to: {self.target_stream.upper()}")
                    print(f"   System will BEEP if contamination is detected!")
                    if stream == 'plastic':
                        print("   Contaminants: METAL and GLASS will trigger alerts")
                    elif stream == 'paper':
                        print("   Contaminants: METAL and GLASS will trigger alerts")
                    elif stream == 'glass':
                        print("   Contaminants: METAL will trigger alerts")
                    elif stream == 'metal':
                        print("   No alerts - you are sorting metal")
                else:
                    print(f" Invalid: {stream}")
                print("-" * 40)
            elif key == ord('c'):
                print(f"\n STATISTICS")
                print(f"   Total Detections: {self.detection_count}")
                print(f"   Total Alerts: {self.alert_count}")
                if self.detection_count > 0:
                    print(f"   Contamination Rate: {(self.alert_count/self.detection_count)*100:.1f}%")
        
        cap.release()
        cv2.destroyAllWindows()
        self.end_session()
        print("\n👋 Session saved to database!")


def main():
    detector = RealTimeDetector()
    detector.run()


if __name__ == "__main__":
    main()
    