#!/bin/bash
echo "Compiling BitMEX System............"
echo "Removing existing file"
rm -r ./trader
echo "File removed, compiling..............."
g++ -o trader trader.cpp bitmex.cpp feed.cpp data.cpp encrypt++/encoder.cpp -lcurl -lcpprest -lcrypto++ -lpthread
echo "BitMEX System Compiled............."
exit 0
