#!/bin/bash
echo "Compiling BitMEX System............"
echo "Removing existing file"
rm -r ./trader
echo "File removed, compiling..............."
g++ -o trader trader.cpp bitmex.cpp feed.cpp data.cpp encrypt++/encoder.cpp -lcurl -lcpprest -lcrypto++ -lpthread -std=c++11 -I/usr/include/python2.7 -lpython2.7
echo "BitMEX System Compiled............."
exit 0
