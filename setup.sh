sudo apt-get update
apt install python3.12-venv
python3 -m venv env
source env/bin/activate
 
apt install xvfb
sudo apt install libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0
sudo apt install libqt5gui5 libqt5core5a libqt5dbus5 libx11-xcb1 libfontconfig1 libfreetype6

pip install -r requirements.txt