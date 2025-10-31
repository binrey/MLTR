sudo cp deploy/template.service /etc/systemd/system/volprof@BTCUSDT.service &&
sudo cp deploy/template.service /etc/systemd/system/volprof@ETHUSDT.service &&

sudo systemctl daemon-reload &&

echo "🚀 Enabling volprof@BTCUSDT.service" &&
sudo systemctl enable volprof@BTCUSDT.service &&

echo "🚀 Enabling volprof@ETHUSDT.service" &&
sudo systemctl enable volprof@ETHUSDT.service