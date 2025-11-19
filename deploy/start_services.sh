echo "🚀 Starting volprof@BTCUSDT.service" &&
sudo systemctl start volprof@BTCUSDT.service &&
sudo systemctl --no-pager status volprof@BTCUSDT.service

echo "🚀 Starting volprof@ETHUSDT.service" &&
sudo systemctl start volprof@ETHUSDT.service &&
sudo systemctl --no-pager status volprof@ETHUSDT.service

echo "🚀 Starting macross@BTCUSDT.service" &&
sudo systemctl start macross@BTCUSDT.service &&
sudo systemctl --no-pager status macross@BTCUSDT.service

echo "🚀 Starting macross@ETHUSDT.service" &&
sudo systemctl start macross@ETHUSDT.service &&
sudo systemctl --no-pager status macross@ETHUSDT.service