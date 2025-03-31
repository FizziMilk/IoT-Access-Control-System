# IoT Access Control System with Dual Authentication via Facial Recognition and OTP 
**This is a BEng Project for LSBU Electronic and Computer Systems Engineering**  

**Technologies and Functionality:**    
  * Amazon AWS EC2 Ubuntu 22.04 Server Backend:  
    * Flask WebApp  
    * SQLAlchemy/SQLite database for data storage and retrieval  
    * MQTT and RESTful API for communication between devices  
    * Free version of Twilio API for OTP generation and verification  
    * BCrypt for password and user information hashing within database  
    * TLS/SSL security over HTTPS and MQTT for secure communication  
    * JWT Access Token Generation for Mobile App Security  
    
  * Raspberry Pi 4b 4GB running Ubuntu 24.04 Desktop Edition    
    * Flask WebApp to provide user interface  
    * Communicates with the backend over secure MQTT and HTTPS(RESTful) messages  
    * Facial Recognition using face_recognition library (in progress)  
    * Manages GPIO to unlock and lock door(Intended for use with a Solenoid Lock)  

  * Expo React Native Mobile Admin Application  
    * Provides administrator interface for door control  
    * Communicates with the Raspberry Pi via Backend with HTTPS RESTful API  
    * Secured via Login and OTP system, JWT Token for session authentication  
    * Provides system administration with functions such as:  
      * Door unlock  
      * Global Door Schedule(Setting times to provide open access)
      * Real Time Access Log inspection for accesses to the door
      * --User Management--  
      * Addition, removal, and management of permitted users via phone number  
      * Management of Access Schedules for individual users  
        
