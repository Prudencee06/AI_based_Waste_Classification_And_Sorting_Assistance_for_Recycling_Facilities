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
        print("PRE-TRAINED YOLO MODEL")
        print("Using YOLOv8n.pt (COCO dataset)")
        print("=" * 50)
        
        from ultralytics import YOLO
        self.model = YOLO('yolov8n.pt')
        print("Pre-trained model loaded - detects bottles, cups, spoons, etc.")
        
        # Session tracking
        self.session_id = None
        self.session_active = False
        self.running = True
        
        # Settings
        self.confidence = 0.3
        self.audio_on = True
        self.target = None
        
        # Stats
        self.detection_count = 0
        self.alert_count = 0
        
        # Colors for waste types
        self.colors = {
            'plastic': (255, 0, 0),
            'metal': (0, 0, 255),
            'paper': (0, 255, 0),
            'glass': (255, 255, 0),
            'cardboard': (0, 165, 255),
        }
        
        # Map YOLO labels to waste types
        self.waste_map = {
            'bottle': 'plastic',
            'cup': 'plastic',
            'bowl': 'plastic',
            'knife': 'metal',
            'fork': 'metal',
            'spoon': 'metal',
            'scissors': 'metal',
            'can': 'metal',
            'book': 'paper',
            'paper': 'paper',
            'cardboard': 'cardboard',
            'glass': 'glass',
            'jar': 'glass',
        }
        
        self.ignore_list = ['person', 'chair', 'table', 'laptop', 'phone', 'tv', 'remote', 'keyboard', 'mouse']
        
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print("\nShutting down...")
        self.running = False
    
    def beep(self):
        """Play a beep sound for contamination alerts"""
        if not self.audio_on:
            return
        
        try:
            winsound.Beep(1200, 600)
            return
        except:
            pass
        
        # Fallback: print bell character
        try:
            print('\a', end='', flush=True)
            return
        except:
            pass
        
        # Last resort: print text
        print("[BEEP]", end='', flush=True)
    
    def test_beep(self):
        print("\nTesting beep...")
        self.beep()
        print(" Beep test complete")
    
    def get_waste_type(self, label):
        label_lower = label.lower()
        if label_lower in self.ignore_list:
            return None
        for key, waste in self.waste_map.items():
            if key in label_lower:
                return waste
        return None
    
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
    
    def log_detection(self, label, confidence, alert, reason):
        if not self.session_active:
            return
        try:
            data = {
                'detected_class': label,
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
        print("PRE-TRAINED YOLO MODEL")
        print("Detects: bottles, cans, spoons, cups, books")
        print("Limitations: May miss paper, glass, cardboard")
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
                    label = self.model.names[cls]
                    waste_type = self.get_waste_type(label)
                    if waste_type:
                        detections.append((waste_type, conf, label, (x1, y1, x2, y2)))
            
            for waste_type, conf, label, (x1, y1, x2, y2) in detections:
                is_alert, alert_msg = self.check_contamination(waste_type)
                
                if is_alert and self.audio_on:
                    self.beep()
                    self.alert_count += 1
                    print(f"\nALERT: {alert_msg}")
                
                self.detection_count += 1
                self.log_detection(waste_type, conf, is_alert, alert_msg)
                
                current_time = time.time()
                if current_time - last_print_time > 1:
                    print(f"  {waste_type.upper()}: {label} ({conf:.0%})")
                    last_print_time = current_time
                
                color = self.colors.get(waste_type, (255, 255, 255))
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
            cv2.putText(frame, f"MODEL: PRE-TRAINED YOLO", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f"Session: {self.session_id}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Target: {self.target or 'NOT SET'}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f"Items: {len(detections)}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.rectangle(frame, (0, h - 25), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, "[q] Quit  [a] Audio  [s] Set Target", (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow('Pre-trained YOLO - Waste Detection', frame)
            
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