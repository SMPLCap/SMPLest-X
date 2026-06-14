#!/usr/bin/env bash
conda create -n smplestx python=3.8 -y
conda activate smplestx
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt

# for osmesa
apt-get update
apt-get install python-opengl -y
apt-get install libosmesa6 -y 

# update version for pyopengl
# Note: please ignore the incompatible error message if 3.1.4 can be installed
pip install pyopengl==3.1.4