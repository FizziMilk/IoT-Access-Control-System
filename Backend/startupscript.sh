#!/usr/bin/env bash

echo Starting flask app on all IPs, port 5000

cd /home/ubuntu/flask-app/Backend

source venv/bin/activate

export FLASK_APP=backend.py

flask run --host=0.0.0.0

