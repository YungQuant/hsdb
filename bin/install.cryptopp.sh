#!/bin/bash

echo "Installing Crypto++"

set -v

git clone https://github.com/weidai11/cryptopp.git --depth 1
cd cryptopp

make
make test
sudo make install

cd ..
rm -rf cryptopp

echo "Finished installing Crypto++"