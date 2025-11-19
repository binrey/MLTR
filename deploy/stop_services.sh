echo "🛑 Stopping volprof@BTCUSDT.service" &&
sudo systemctl stop volprof@BTCUSDT.service &&
sudo systemctl --no-pager status volprof@BTCUSDT.service

echo "🛑 Stopping volprof@ETHUSDT.service" &&
sudo systemctl stop volprof@ETHUSDT.service &&
sudo systemctl --no-pager status volprof@ETHUSDT.service

echo "🛑 Stopping macross@BTCUSDT.service" &&
sudo systemctl stop macross@BTCUSDT.service &&
sudo systemctl --no-pager status macross@BTCUSDT.service

echo "🛑 Stopping macross@ETHUSDT.service" &&
sudo systemctl stop macross@ETHUSDT.service &&
sudo systemctl --no-pager status macross@ETHUSDT.service