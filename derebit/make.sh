#!/bin/bash
g++ -o algo test.cpp deribit.cpp greeks/greeks.cpp wsocket.cpp encoder.cpp database.cpp -std=c++11 -lcurl -lpthread -lcpprest -I/usr/include/python2.7 -lpython2.7
echo "------------------------------------"
echo " Options trading system has compiled"
echo "           ..!. (*_*).!..           "
echo "               MoAlgo$              "
echo "                 /\                 "
echo "                /  \                "
echo "               /(<>)\               "
echo "              /______\              "
echo "                                    "
exit 0
