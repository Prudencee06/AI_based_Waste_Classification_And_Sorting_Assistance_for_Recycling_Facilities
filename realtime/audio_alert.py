import time
import sys

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    winsound = None  
    HAS_WINSOUND = False
    print(" winsound not available (Windows only). Audio alerts will use console beeps.")


class AudioAlert:    
    def __init__(self, frequency=1000, duration=200):
        self.frequency = frequency
        self.duration = duration
        self.enabled = True
        self.last_alert_time = 0
        self.alert_cooldown = 1  # seconds between alerts
    
    def beep(self):
        if not self.enabled:
            return
        
        # Check cooldown to avoid constant beeping
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.last_alert_time = current_time
        
        if HAS_WINSOUND and winsound is not None:
            try:
                winsound.Beep(self.frequency, self.duration)
            except Exception as e:
                print(f" Error playing beep: {e}")
        else:
            # Fallback: print alert to console (ASCII bell)
            print("\a", end='', flush=True)
            print(" BEEP! (Console alert - no sound card detected)")
    
    def alert(self, message=None):
        self.beep()
        if message:
            print(f"!!!! {message}")
    
    def alert_sequence(self, repetitions=3, delay=0.2):
        for i in range(repetitions):
            self.beep()
            time.sleep(delay)
    
    def enable(self):
        self.enabled = True
        print(" Audio alerts enabled")
    
    def disable(self):
        self.enabled = False
        print(" Audio alerts disabled")
    
    def toggle(self):
        self.enabled = not self.enabled
        status = "ON" if self.enabled else "OFF"
        print(f" Audio alerts: {status}")
        return self.enabled


# For testing
if __name__ == "__main__":
    print("Testing AudioAlert...")
    alert = AudioAlert()
    
    print("Playing test beep...")
    alert.beep()
    
    print("Playing alert sequence...")
    alert.alert_sequence(repetitions=2)
    
    print("\n Audio alert test complete!")