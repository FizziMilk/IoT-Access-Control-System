#!/bin/bash

# Navigate to the backend directory
cd "$(dirname "$0")"

# Ensure virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install dependencies (without numpy)
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the backend application
echo "Starting Flask backend..."
python backend.py 