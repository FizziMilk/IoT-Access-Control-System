import RPi.GPIO as GPIO
from threading import Timer

class DoorController:
    def __init__(self, door_pin=17):
        self.door_pin = door_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.door_pin, GPIO.OUT)

    def unlock_door(self, duration=10):
        print("[DEBUG] Unlocking door...")
        GPIO.output(self.door_pin, GPIO.HIGH)  # Activate door relay
        print("[DEBUG] Door unlocked")

        # Schedule the door to lock after the specified duration
        Timer(duration, self.lock_door).start()

    def lock_door(self):
        GPIO.output(self.door_pin, GPIO.LOW)  # Deactivate door relay
        print("[DEBUG] Door locked")

    def cleanup(self):
        GPIO.cleanup()
        print("[DEBUG] GPIO cleanup completed") 