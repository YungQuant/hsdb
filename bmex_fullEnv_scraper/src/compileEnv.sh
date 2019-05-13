#!/bin/bash
echo "Compiling BitMEX ENV System............"
echo "Removing existing file"
rm -r ./envtrader
echo "File removed, compiling..............."
g++ -o envtrader trader.cpp CLIENT/REST/bitmex.cpp CLIENT/feed.cpp CLIENT/REST/DATA/data.cpp CLIENT/REST/AUTH/encoder.cpp -lcurl -lcpprest -lcrypto++ -lpthread -std=c++14 -I/usr/include/python2.7 -lpython2.7
echo "BitMEX ENV System Compiled............."
exit 0
