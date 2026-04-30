#!/bin/bash
set -e

echo "TrueVision Pi Setup"

sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
sudo apt install -y libasound2-dev portaudio19-dev
sudo apt install -y gstreamer1.0-plugins-bad gstreamer1.0-plugins-good
sudo apt install -y libjpeg-dev zlib1g-dev libfreetype6-dev liblcms2-dev libopenjp2-7-dev libtiff5-dev libxcb1-dev

if [ ! -d "venv" ]; then
    python3 -m venv venv --system-site-packages
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-pi.txt

mkdir -p models
cd models
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
fi
if [ ! -f "dlib_face_recognition_resnet_model_v1.dat" ]; then
    wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2
fi
cd ..

echo "Setting up UART (ttyAMA0)"
if ! grep -q "dtoverlay=uart0" /boot/firmware/config.txt; then
    echo "dtoverlay=uart0" | sudo tee -a /boot/firmware/config.txt
fi

sudo systemctl disable serial-getty@ttyAMA0.service || true
sudo adduser $USER dialout

echo "Setup complete. Please reboot."
