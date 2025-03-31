# IoT Access Control System with Dual Authentication via Facial Recognition and OTP 
**This is a BEng Project for LSBU Electronic and Computer Systems Engineering**  

**Technologies and Functionality:**    
  * **Amazon AWS EC2 Ubuntu 22.04 Server Backend:**    
    * Flask 3.10.0 WebApp
    * SQLAlchemy 2.0.40/SQLite database for data storage and retrieval  
    * MQTT and RESTful API for communication between devices  
    * Free version of Twilio API for OTP generation and verification  
    * BCrypt for password and user information hashing within database  
    * TLS/SSL security over HTTPS and MQTT for secure communication  
    * JWT Access Token Generation for Mobile App Security  
    
  * **Raspberry Pi 4b 4GB running Ubuntu 24.04 Desktop Edition**      
    * Flask 3.10.0 WebApp to provide user interface  
    * Communicates with the backend over secure MQTT and HTTPS(RESTful) messages  
    * Facial Recognition using face_recognition library (in progress)  
    * Manages GPIO to unlock and lock door(Intended for use with a Solenoid Lock)  

  * **Expo React Native Mobile Admin Application**
    * Expo Version 52.0.37
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
     
   
Deployment Steps:  

Create EC2 instance and setup networking settings to allow HTTPS communication  
 * Install all dependencies(Based on import, might make an import file later)
 * Create .env file to house Twilio tokens, mqtt_broker url, and JWT secret key
 * Create certificates on local computer, transfer to EC2, add to system trusted certificate store
 * (Include IP address and Hostname of EC2 instance in SAN field of certificates)
 * Setup nginx to route all traffic to HTTPS in /etc/nginx/sites-available/backend
 * Setup mosquitto, store certificates inside /etc/mosquitto/certs, setup conf.d file for TLS, point to certs
 * run startupscript.sh with Bash, and/or add to crontab -e with @reboot command
Set up Ubuntu on Raspberry Pi(Requires atleast 2GB of RAM for functional Facial Recognition)
 * Install all dependencies
 * Copy certificate from local computer to Raspberry Pi and add to trusted certificate store
 * Set up mosquitto conf.d, set up nginx 
 * run main.py or set up autorun
 * navigate to localhost:5000 for User Interface
Set up Expo mobile app, currently with the npx expo start command (Still development mode)
 * Install all dependencies
 * create .env file with Backend IP
