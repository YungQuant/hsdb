#!/bin/bash

echo "Installing OpenSSL"

set -v

git clone https://github.com/openssl/openssl.git --depth 1
cd openssl

./config
make
make test
sudo make install

cd ..
rm -rf openssl

echo "Finished installing OpenSSL"