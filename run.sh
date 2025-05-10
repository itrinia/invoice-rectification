#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check versions
echo "Checking package versions..."
python check_versions.py

# Run setup script to download and verify model
echo "Setting up model..."
python setup.py

# Run Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py 