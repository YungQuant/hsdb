#!/bin/bash
echo "Compiling BitMEX System............"
echo "Removing existing file"
rm -r ./trader
echo "File removed, compiling..............."
g++ -o trader trader.cpp CLIENT/REST/bitmex.cpp CLIENT/feed.cpp CLIENT/REST/DATA/data.cpp CLIENT/REST/AUTH/encoder.cpp -lcurl -lcpprest -lcrypto++ -lpthread -std=c++14 -I/usr/include/python2.7 -lpython2.7
echo "BitMEX System Compiled............."
echo "Running BitMEX Visualizer..........."
./trader
exit 0
