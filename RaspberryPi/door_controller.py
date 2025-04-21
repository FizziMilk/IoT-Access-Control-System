import RPi.GPIO as GPIO
from threading import Timer

class DoorController:
    def __init__(self, door_pin=17):
        self.door_pin = door_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.door_pin, GPIO.OUT)

    def unlock_door(self, duration=10):
        print("[DEBUG] Unlocking door...")
        try:
            GPIO.output(self.door_pin, GPIO.HIGH)  # Activate door relay
            print("[DEBUG] Door unlocked - GPIO pin set to HIGH")
            
            # Schedule the door to lock after the specified duration
            print(f"[DEBUG] Scheduling door to lock after {duration} seconds")
            Timer(duration, self.lock_door).start()
            print(f"[DEBUG] Lock timer started")
        except Exception as e:
            print(f"[ERROR] Failed to unlock door: {str(e)}")

    def lock_door(self):
        try:
            GPIO.output(self.door_pin, GPIO.LOW)  # Deactivate door relay
            print("[DEBUG] Door locked - GPIO pin set to LOW")
        except Exception as e:
            print(f"[ERROR] Failed to lock door: {str(e)}")

    def cleanup(self):
        GPIO.cleanup()
        print("[DEBUG] GPIO cleanup completed") 