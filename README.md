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

This system provides a secure, multi-factor door access control solution using IoT devices, mobile applications, and facial recognition. The system consists of three main components:
1. Backend Server (EC2)
2. Raspberry Pi Controller
3. Mobile Application

## Backend Server Setup (EC2)

### 1. Create and Configure EC2 Instance
```bash
# Allow inbound traffic for:
- HTTPS (443)
- MQTT over TLS (8883)
- SSH (22)
```

### 2. Install Dependencies
```bash
# Update package list
sudo apt update
sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip -y

# Install required Python packages
pip3 install flask flask-restful flask-sqlalchemy flask-mqtt twilio python-dotenv flask-bcrypt flask-jwt-extended

# Install Nginx and Mosquitto
sudo apt install nginx mosquitto mosquitto-clients -y
```

### 3. Configure Environment Variables
Create a `.env` file in the Backend directory:
```bash
# Backend/.env
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
MQTT_BROKER_URL=your_ec2_public_ip
JWT_SECRET_KEY=your_secret_key
```

### 4. SSL/TLS Certificate Setup
```bash
# On your local machine, generate certificates
openssl req -new -x509 -days 365 -extensions v3_ca -keyout ca.key -out ca.crt
openssl genrsa -out server.key 2048
openssl req -new -out server.csr -key server.key -config openssl.cnf  # Include EC2 IP and hostname in SAN
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 -extensions v3_req -extfile openssl.cnf

# Transfer certificates to EC2
scp ca.crt server.crt server.key ubuntu@your-ec2-ip:~/certs/

# On EC2: Add to trusted certificates
sudo cp ca.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

### 5. Configure Nginx
Create `/etc/nginx/sites-available/backend`:
```nginx
server {
    listen 443 ssl;
    server_name your_ec2_ip;

    ssl_certificate /path/to/server.crt;
    ssl_certificate_key /path/to/server.key;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
```bash
# Enable the site and restart Nginx
sudo ln -s /etc/nginx/sites-available/backend /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. Configure Mosquitto
Create `/etc/mosquitto/conf.d/default.conf`:
```
listener 8883
cafile /etc/mosquitto/ca_certificates/ca.crt
certfile /etc/mosquitto/certs/server.crt
keyfile /etc/mosquitto/certs/server.key
require_certificate true
tls_version tlsv1.2
```
```bash
# Copy certificates
sudo cp ca.crt /etc/mosquitto/ca_certificates/
sudo cp server.crt server.key /etc/mosquitto/certs/
sudo chmod 644 /etc/mosquitto/ca_certificates/ca.crt
sudo chmod 644 /etc/mosquitto/certs/server.crt
sudo chmod 600 /etc/mosquitto/certs/server.key

# Restart Mosquitto
sudo systemctl restart mosquitto
```

### 7. Create Admin User
```python
# add_user.py
from backend import Admin, db
admin = Admin(username="admin", phone_number="+1234567890")
admin.set_password("your_password")
db.session.add(admin)
db.session.commit()
```

### 8. Setup Auto-start
```bash
# Create startup script
echo '#!/bin/bash
cd /path/to/backend
python3 backend.py' > startup.sh

# Make it executable
chmod +x startup.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "@reboot /path/to/startup.sh") | crontab -
```

## Raspberry Pi Setup

### 1. Install Ubuntu Server
Download and install Ubuntu Server 22.04 LTS (64-bit) for Raspberry Pi (minimum 2GB RAM)

### 2. Install Dependencies
```bash
sudo apt update
sudo apt upgrade -y

# Install required packages
sudo apt install python3-pip python3-opencv nginx -y

# Install Python packages
pip3 install flask flask-mqtt opencv-python-headless face-recognition dlib numpy
```

### 3. Configure Environment Variables
Create `.env` in the RaspberryPi directory:
```bash
BACKEND_URL=https://your_ec2_ip
MQTT_BROKER=your_ec2_ip
MQTT_PORT=8883
CA_CERT=/path/to/ca.crt
```

### 4. Copy Certificates
```bash
# Copy CA certificate from local machine
scp ca.crt ubuntu@raspberry-pi-ip:~/certs/
sudo cp ca.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

### 5. Configure Nginx
Create `/etc/nginx/sites-available/rpi`:
```nginx
server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
```bash
sudo ln -s /etc/nginx/sites-available/rpi /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. Setup Auto-start
```bash
# Create startup script
echo '#!/bin/bash
cd /path/to/raspberrypi
python3 main.py' > startup.sh

# Make it executable
chmod +x startup.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "@reboot /path/to/startup.sh") | crontab -
```

## Mobile App Setup

### 1. Install Dependencies
```bash
# Install Node.js and npm
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs

# Install Expo CLI
npm install -g expo-cli

# Install project dependencies
cd MobileApp
npm install
```

### 2. Configure Environment
Create `.env` in the MobileApp directory:
```bash
EXPO_PUBLIC_BACKEND_IP=https://your_ec2_ip
```

### 3. Run the App
```bash
# Development
npx expo start

# Build for production
eas build --platform android  # For Android
eas build --platform ios      # For iOS
```

## Security Notes
- Keep all `.env` files secure and never commit them to version control
- Regularly update SSL certificates before expiry
- Keep all systems updated with security patches
- Monitor access logs regularly
- Backup the SQLite database periodically

## Troubleshooting
- Check Nginx error logs: `sudo tail -f /var/log/nginx/error.log`
- Check Mosquitto logs: `sudo tail -f /var/log/mosquitto/mosquitto.log`
- Check application logs: `tail -f backend.log`
- Verify MQTT connectivity: `mosquitto_sub -h your_ec2_ip -p 8883 --cafile ca.crt -t "test/topic"` 
