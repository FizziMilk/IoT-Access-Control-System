#!/usr/bin/env python3
from face_system import FaceRecognitionSystem

def main():
    # Create face recognition system
    face_system = FaceRecognitionSystem()
    
    # Settings
    use_liveness = True
    
    while True:
        print("\nFace Recognition System")
        print("----------------------")
        print("1. Register new user")
        print("2. Recognize face")
        print("3. List registered users")
        print("4. Delete user")
        print("5. Generate encoding for backend")
        print(f"6. Toggle liveness detection (currently: {'ON' if use_liveness else 'OFF'})")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-6): ")
        
        if choice == "1":
            face_system.register_new_user(use_liveness=use_liveness)
        elif choice == "2":
            results = face_system.recognize_face(use_liveness=use_liveness)
            if results:
                print("\nRecognition Results:")
                for i, result in enumerate(results):
                    print(f"Face {i+1}: {result['name']} (Confidence: {result['confidence']:.2f})")
        elif choice == "3":
            face_system.list_users()
        elif choice == "4":
            face_system.delete_user()
        elif choice == "5":
            encoding = face_system.get_encoding_for_backend()
            if encoding:
                print(f"Generated encoding of length {len(encoding)}")
                print("You can send this encoding to your backend API")
        elif choice == "6":
            use_liveness = not use_liveness
            print(f"Liveness detection turned {'ON' if use_liveness else 'OFF'}")
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 