import RPi.GPIO as GPIO
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class DoorController:
    def __init__(self, relay_pin=17):
        """Initialize the door controller with GPIO settings."""
        self.relay_pin = relay_pin
        self.schedule_file = 'door_schedule.json'
        self.schedule = self.load_schedule()
        self.unlock_callback = None
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.relay_pin, GPIO.OUT)
        GPIO.output(self.relay_pin, GPIO.HIGH)  # Door is locked by default
        
        logger.info(f"Door controller initialized on pin {relay_pin}")

    def set_unlock_callback(self, callback):
        """Set the callback function to be called when the door is unlocked."""
        self.unlock_callback = callback
        logger.info("Unlock callback set")

    def load_schedule(self):
        """Load the door schedule from file."""
        try:
            if os.path.exists(self.schedule_file):
                with open(self.schedule_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading schedule: {str(e)}")
            return {}

    def save_schedule(self):
        """Save the door schedule to file."""
        try:
            with open(self.schedule_file, 'w') as f:
                json.dump(self.schedule, f)
        except Exception as e:
            logger.error(f"Error saving schedule: {str(e)}")

    def check_door_status(self):
        """Check if the door should be unlocked based on schedule."""
        current_time = datetime.now()
        current_day = current_time.strftime('%A').lower()
        current_time_str = current_time.strftime('%H:%M')

        if current_day in self.schedule:
            day_schedule = self.schedule[current_day]
            for time_slot in day_schedule:
                if time_slot['start'] <= current_time_str <= time_slot['end']:
                    return {
                        'is_scheduled': True,
                        'message': f"Door is scheduled to be unlocked from {time_slot['start']} to {time_slot['end']}"
                    }

        return {
            'is_scheduled': False,
            'message': "Door is not scheduled to be unlocked"
        }

    def unlock_door(self, reason, phone_number):
        """Unlock the door for a specified duration."""
        try:
            # Log the unlock attempt
            logger.info(f"Unlocking door - Reason: {reason}, Phone: {phone_number}")
            
            # Activate the relay (LOW signal to unlock)
            GPIO.output(self.relay_pin, GPIO.LOW)
            
            # Call the unlock callback if set
            if self.unlock_callback:
                self.unlock_callback(phone_number, reason)
            
            # Wait for 5 seconds
            import time
            time.sleep(5)
            
            # Lock the door again (HIGH signal to lock)
            GPIO.output(self.relay_pin, GPIO.HIGH)
            
            logger.info("Door locked after 5 seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error unlocking door: {str(e)}")
            return False

    def cleanup(self):
        """Cleanup GPIO resources."""
        try:
            GPIO.cleanup()
            logger.info("GPIO cleanup completed")
        except Exception as e:
            logger.error(f"Error during GPIO cleanup: {str(e)}") 