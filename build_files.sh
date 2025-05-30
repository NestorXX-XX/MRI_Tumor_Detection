#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Collect static files
python Tumor_Detection_APP/manage.py collectstatic --noinput

# Create necessary directories
mkdir -p staticfiles
mkdir -p media 