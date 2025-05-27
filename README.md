# IoT Access Control System 
BEng Project For LSBU Electronic and Computer Systems Engineering

A full‑stack IoT door access control project combining secure backend services, edge‑device facial recognition, and a mobile administration app.
Installation of this project from GitHub has not yet been attempted, if you have any issues get in touch.

![image](https://github.com/user-attachments/assets/fc46b3c4-89ce-4ecd-9f65-a963463834b9)


---
## Table of Contents
- [Overview](#overview)
- [Architectural Components](#architectural-components)
- [Software & Hardware Requirements](#software--hardware-requirements)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

---
## Overview
This system provides multi‑factor door access control via:

1. **Backend Server** – Flask REST API & MQTT broker, hosted on AWS EC2. Twilio API for OTP handling.
2. **Edge Device** – Raspberry Pi client running face recognition + liveness detection, communicating over HTTPS & secure MQTT.
3. **Mobile App** – Expo React Native app for administrators to manage users, schedules, and door locks.

Security features include OTP via Twilio, JWT‑authenticated endpoints, mutual TLS for MQTT, and HTTPS for all REST traffic.

---
## Architectural Components
- **RESTful API** for user, OTP, schedule, and access-logs management.
- **MQTT Broker** for real‑time door control commands and status updates.
- **Facial Recognition & Liveness** on Pi using OpenCV, dlib, and custom LBP+reflection analysis.
- **React Native App** for cross‑platform (iOS/Android) administration with JWT login & OTP verification.
- **Nginx + Let's Encrypt** for HTTPS termination and static page hosting.

---
## Software & Hardware Requirements
<details>
<summary><strong>🖥️ Backend (Server)</strong></summary>

- Operating System: Ubuntu 22.04 LTS (Jammy Jellyfish)
- Python: 3.10.x
- Flask: 2.2.3
- Flask‑Bcrypt: 1.0.1
- Flask‑CORS: 3.0.10
- Flask‑JWT‑Extended: 4.4.4
- Flask‑MQTT: 1.1.1
- Flask‑RESTful: 0.3.9
- Flask‑SQLAlchemy: 3.0.3
- python‑dotenv: 1.0.0
- SQLAlchemy: 2.0.4
- Twilio: 7.16.4
- Requests: 2.28.2
- Nginx: latest (via apt)
- Mosquitto (MQTT): latest (via apt)
- SQLite: embedded local database for Flask-SQLAlchemy

</details>

<details>
<summary><strong>📟 Raspberry Pi</strong></summary>

- Hardware: Raspberry Pi 4 Model B (4GB RAM)
- Operating System: Ubuntu 24.04 LTS for arm64/Desktop
- Python: 3.10.x
- OpenCV (headless): opencv‑python‑headless
- face_recognition (dlib dependency)
- dlib: latest
- numpy: latest
- Flask: 2.2.3 (for Pi‑hosted UI)
- MQTT Handler: paho‑mosquitto via Flask‑MQTT

</details>

<details>
<summary><strong>📱 Mobile App (React Native)</strong></summary>

- Node.js: >=16.x (LTS recommended)
- npm: >=8.x
- Expo CLI: ~6.x
- Expo SDK: ~52.0.37
- React Native: 0.76.7
- Dependencies (see `MobileApp/package.json`):
  - @expo/vector-icons
  - expo‑router, expo‑secure-store, expo‑splash‑screen, expo‑constants, etc.
  - react‑native‑paper, react‑native‑reanimated, @react-navigation/*
  - react‑native-webview, async‑storage, datetimepicker, etc.

</details>

---
## Installation & Setup

<details>
<summary><strong>Backend Server</strong></summary>

```bash
# 1.1 Clone the repo
git clone <your-repo-url>
cd IoT-Access-Control-System/Backend

# 1.2 Update and install system packages
sudo apt update && sudo apt install -y python3 python3-venv python3-pip nginx mosquitto mosquitto-clients

# 1.3 Create Python virtual environment
env=venv && python3 -m venv $env && source $env/bin/activate

# 1.4 Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 1.5 Configure environment variables
cp .env.example .env  # edit .env to add your own TWILIO, MQTT, JWT settings

# 1.6 Start and enable the backend service
./start_backend.sh
```

</details>


<details>
<summary><strong>Raspberry Pi </strong></summary>

```bash
# 2.1 Prepare the Pi (Ubuntu 24.04)
sudo apt update && sudo apt install -y python3 python3-venv python3-pip python3-opencv mosquitto mosquitto-clients nginx

# 2.2 Clone and enter the Pi client directory
git clone <your-repo-url>
cd IoT-Access-Control-System/RaspberryPi

# 2.3 Create virtual environment and install dependencies
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2.4 Configure certificates for MQTT and HTTPS
cp .env.example .env  # update BACKEND_URL, MQTT, cert paths
sudo cp $CA_CERT_PATH /usr/local/share/ca-certificates/pi-ca.crt
sudo update-ca-certificates

# 2.5 Start the Pi client
python3 main.py
```

</details>


<details>
<summary><strong>Mobile App (Expo)</strong></summary>

```bash
# 3.1 Prerequisites
# Node.js >= 16.x, npm >= 8.x

# 3.2 Install Expo CLI globally (if not already)
npm install -g expo-cli

# 3.3 Clone and install dependencies
git clone <your-repo-url>
cd IoT-Access-Control-System/MobileApp
npm install

# 3.4 Configure environment
cp .env.example .env  # set EXPO_PUBLIC_BACKEND_IP=https://your.domain/api

# 3.5 Launch in development
npm start  # then scan QR code in Expo Go
```

</details>

---
## Configuration
- **Example env files**: Copy the `.env.example` in each folder (Backend, RaspberryPi, MobileApp) to `.env` and fill in your own values to configure secrets and URLs.
- **Nginx**: Configure `/etc/nginx/sites-available/your-domain.conf` to proxy `/api/` to Flask on port 5000.
- **Certificates**: Use Let's Encrypt for TLS; sync fullchain.pem + privkey.pem into Nginx and Pi's trusted CA store.
- **MQTT**: Use port 8883 with mutual TLS (configure mosquitto to require certificates, see `mosquitto.conf`).

---
## Usage
1. API endpoints under `https://your.domain/api/` (login, verify-otp, users, schedule).
2. Mobile App: login → OTP → manage users & schedule → unlock door.
3. Pi Client: continuously runs face recognition + liveness → calls `/api/verify-face` → activates GPIO.

---
## Troubleshooting
- **403 Forbidden**: Confirm your Nginx `root` path matches `/var/www/your-site` and that files/folders are owned by `www-data:www-data` with `755` for directories and `644` for files.
- **API 404 or 301 redirects**: Verify Nginx `location /api/` block exists and proxy_pass is set to `http://127.0.0.1:5000` (or your Flask host).
- **TLS/SSL errors**: Check that Let's Encrypt certs (`fullchain.pem`/`privkey.pem`) paths in Nginx match your domain, and renew using `certbot renew` if expired.
- **MQTT connection failures**: Ensure Mosquitto is running on port `8883`, client certificates are valid, and the Pi's env vars point to correct `MQTT_BROKER_URL` and `PORT`.
- **Camera not found or busy**: On Raspberry Pi, ensure no other process (like another OpenCV instance) is holding the camera; reboot or run `sudo pkill -f python` to clear.
- **Face detection issues**: Confirm `dlib` and `face_recognition` installed correctly in the venv; review `/tmp/face_recognition_process.log` for errors.
- **Mobile App build or runtime errors**: Clear caches with `expo start --clear`, remove `node_modules` and reinstall, and verify `EXPO_PUBLIC_BACKEND_IP` uses `https://` and includes `/api` if needed.
- **General logs**: Check system logs:
  - Nginx: `/var/log/nginx/error.log`
  - Mosquitto: `/var/log/mosquitto/mosquitto.log`
  - Flask backend: logs in `instance/` or console output
  - Pi client: console output or stored logs in `/tmp`
