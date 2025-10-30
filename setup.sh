sudo apt-get update
sudo apt install -y python3.12-venv
python3 -m venv venv
source env/bin/activate
 
sudo apt install -y xvfb
sudo apt install -y libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0
sudo apt install -y libqt5gui5 libqt5core5a libqt5dbus5 libx11-xcb1 libfontconfig1 libfreetype6

pip install -r requirements.txt