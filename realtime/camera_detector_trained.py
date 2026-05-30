import cv2
import requests
import time
import sys
import os
import signal
import winsound

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RealTimeDetector:
    def __init__(self, api_base_url="http://127.0.0.1:8000/api"):
        self.api_base_url = api_base_url
        
        print("=" * 50)
        print("CUSTOM TRAINED YOLO MODEL")
        print("=" * 50)
        
        from ultralytics import YOLO
        
        # Path to your custom trained model (correct path)
        model_path = r'C:\AI-Based_Waste_Classification_and_Sorting_Assistance_System_for_Recycling_Facilities\yolo_waste\best.pt'
        
        # Another path 
        if not os.path.exists(model_path):
            model_path = r'C:\AI-Based_Waste_Classification_and_Sorting_Assistance_System_for_Recycling_Facilities\runs\detect\trained_model\waste_detector\weights\best.pt'
        
        try:
            self.model = YOLO(model_path)
            self.class_names = {
                0: 'biodegradable',
                1: 'cardboard', 
                2: 'glass',
                3: 'metal',
                4: 'paper',
                5: 'plastic'
            }
            print(f"Custom model loaded from: {model_path}")
            print(f"Classes: {list(self.class_names.values())}")
        except Exception as e:
            print(f"Error loading custom model: {e}")
            print("Falling back to pre-trained model")
            self.model = YOLO('yolov8n.pt')
            self.class_names = {0: 'plastic', 1: 'metal', 2: 'paper', 3: 'glass', 4: 'cardboard'}
        
        # Session tracking
        self.session_id = None
        self.session_active = False
        self.running = True
        
        # Settings
        self.confidence = 0.4
        self.audio_on = True
        self.target = None
        
        # Stats
        self.detection_count = 0
        self.alert_count = 0
        
        self.colors = {
            'plastic': (255, 0, 0),        
            'metal': (0, 0, 255),          
            'paper': (0, 255, 0),          
            'glass': (255, 255, 0),        
            'cardboard': (0, 165, 255),    
            'biodegradable': (255, 255, 255),  
            'default': (200, 200, 200)
        }
        
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print("\nShutting down...")
        self.running = False
    
    def beep(self):
        try:
            winsound.Beep(1200, 600)
        except:
            try:
                print('\a', end='', flush=True)
            except:
                print("[BEEP]", end='', flush=True)
    
    def test_beep(self):
        print("\nTesting beep...")
        self.beep()
        print(" Beep test complete")
    
    def get_waste_type(self, class_id):
        """Get waste type from class ID"""
        if class_id in self.class_names:
            return self.class_names[class_id]
        return "unknown"
    
    def start_session(self, operator=None, shift=None):
        try:
            if not shift:
                hour = time.localtime().tm_hour
                if 6 <= hour < 14:
                    shift = 'morning'
                elif 14 <= hour < 22:
                    shift = 'afternoon'
                else:
                    shift = 'night'
            
            if not operator:
                operator = 'AI Sorter'
            
            response = requests.post(
                f"{self.api_base_url}/sessions/",
                json={'operator_name': operator, 'shift': shift},
                timeout=5
            )
            
            if response.status_code == 201:
                data = response.json()
                self.session_id = data['id']
                self.session_active = True
                print(f"\nSession started: {self.session_id}")
                return True
            return False
        except Exception as e:
            print(f"Session error: {e}")
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
                print(f"\nSummary: {data['total_detections']} detections, {data['total_alerts']} alerts")
        except:
            pass
        self.session_active = False
    
    def log_detection(self, waste_type, confidence, alert, reason):
        if not self.session_active:
            return
        try:
            data = {
                'detected_class': waste_type,
                'confidence_score': confidence,
                'alert_triggered': alert,
                'alert_reason': reason,
                'session_id': str(self.session_id)
            }
            requests.post(f"{self.api_base_url}/detections/", json=data, timeout=3)
        except:
            pass
    
    def check_contamination(self, waste_type):
        if not self.target:
            return False, None
        if waste_type == 'metal' and self.target != 'metal':
            return True, f"METAL in {self.target.upper()} stream"
        if waste_type == 'glass' and self.target in ['plastic', 'paper']:
            return True, f"GLASS in {self.target.upper()} stream"
        return False, None
    
    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        print("\n" + "=" * 50)
        print("CUSTOM TRAINED YOLO MODEL")
        print("Classes: biodegradable, cardboard, glass, metal, paper, plastic")
        print("Note: Paper and plastic have lower accuracy due to limited training data")
        print("=" * 50)
        
        operator = input("Operator (Enter for default): ").strip()
        shift = input("Shift (morning/afternoon/night): ").strip()
        
        if not self.start_session(operator if operator else None, shift if shift else None):
            cap.release()
            return
        
        print("\nControls: [s] Target  [a] Audio  [q] Quit\n")
        
        frame_count = 0
        start = time.time()
        last_print_time = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 2 == 0:
                continue
            
            results = self.model(frame, conf=self.confidence, verbose=False)
            
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    waste_type = self.get_waste_type(cls)
                    if waste_type and waste_type != "unknown":
                        detections.append((waste_type, conf, (x1, y1, x2, y2)))
            
            for waste_type, conf, (x1, y1, x2, y2) in detections:
                is_alert, alert_msg = self.check_contamination(waste_type)
                
                if is_alert and self.audio_on:
                    self.beep()
                    self.alert_count += 1
                    print(f"\nALERT: {alert_msg}")
                
                self.detection_count += 1
                self.log_detection(waste_type, conf, is_alert, alert_msg)
                
                current_time = time.time()
                if current_time - last_print_time > 1:
                    print(f"  {waste_type.upper()}: {conf:.0%}")
                    last_print_time = current_time
                
                color = self.colors.get(waste_type, self.colors['default'])
                if is_alert:
                    color = (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{waste_type.upper()} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw + 5, y1), color, -1)
                cv2.putText(frame, label_text, (x1 + 2, y1 - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # HUD
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (300, 70), (0, 0, 0), -1)
            fps = frame_count / (time.time() - start) if time.time() - start > 0 else 0
            cv2.putText(frame, f"MODEL: CUSTOM TRAINED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame, f"Session: {self.session_id}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Target: {self.target or 'NOT SET'}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f"Items: {len(detections)}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.rectangle(frame, (0, h - 25), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, "[q] Quit  [a] Audio  [s] Set Target", (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow('Custom Trained YOLO - Waste Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('a'):
                self.audio_on = not self.audio_on
                status = "ON" if self.audio_on else "OFF"
                print(f"\nAudio alerts: {status}")
                if self.audio_on:
                    self.test_beep()
            elif key == ord('s'):
                new_target = input("\nSet target (plastic/paper/glass): ").lower()
                if new_target in ['plastic', 'paper', 'glass']:
                    self.target = new_target
                    print(f"Target: {self.target.upper()}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.end_session()
        print("\nSystem stopped.")


def main():
    detector = RealTimeDetector()
    detector.run()


if __name__ == "__main__":
    main()