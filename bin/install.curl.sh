#!/bin/bash

echo "Installing Curl"

set -v

git clone https://github.com/curl/curl.git --depth 1
cd curl

./buildconf
./configure
make
make test
sudo make install

cd ..
rm -rf curl

echo "Finished installing Curl"