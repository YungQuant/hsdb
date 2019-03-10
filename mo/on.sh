#!/bin/bash
echo "Compiling bitmex trading system"
g++ -o algo test.cpp bitmex.cpp compute.cpp encrypt++/encoder.cpp -lcrypto++ -lcurl -lpthread -lcpprest
echo "System has compiled"
exit 0
