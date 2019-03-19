BitMEX Trader Instructions

You need to have [curl, crypto++, cpprest, boost] installed, so run

./install.sh


Compile the program by just running the shell script

./run.sh


Run the program with 

./trader

This program is able to run a websocket in parallel to a strategy script.
Several authenticated and regular feeds have been added into the subscription
list. Additionally there is a working rest api capable of placing and cancelling
orders.
