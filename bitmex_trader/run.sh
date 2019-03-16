#!/bin/bash
echo "Compiling BitMEX System............"
g++ -o trader trader.cpp feed.cpp -lcpprest -lcrypto++ -lpthread
echo "BitMEX System Compiled............."
exit 0
