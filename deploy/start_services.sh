echo "🚀 Starting volprof@BTCUSDT.service" &&
sudo systemctl start volprof@BTCUSDT.service &&
sudo systemctl --no-pager status volprof@BTCUSDT.service

echo "🚀 Starting volprof@ETHUSDT.service" &&
sudo systemctl start volprof@ETHUSDT.service &&
sudo systemctl --no-pager status volprof@ETHUSDT.service