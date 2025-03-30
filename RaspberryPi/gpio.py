import RPi.GPIO as GPIO
from threading import Timer

DOOR_PIN = 17

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(DOOR_PIN, GPIO.OUT)

def unlock_door(duration=10):
    GPIO.output(DOOR_PIN, GPIO.HIGH)
    print("[DEBUG] Door unlocked")
    Timer(duration, lock_door).start()

def lock_door():
    GPIO.output(DOOR_PIN, GPIO.LOW)
    print("[DEBUG] Door locked")

def cleanup_gpio():
    GPIO.cleanup()
    print("[DEBUG] GPIO cleanup completed")