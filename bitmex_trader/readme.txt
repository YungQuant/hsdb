BitMEX Trader Instructions

You need to have these installed for this to run

libcpprest-dev
libcrypto++-dev
libcurl4-openssl-dev

Compile the program by just running the shell script

./run.sh


Run the program with 

./trade

As of now, this program subscribes to some authenticated and data
streams with the websocket and prints the output out. There is
a lot of data we are working with because I queried it for all
coins on there.
