#!/bin/bash

echo "Installing prereqs"

set -v

git clone --recurse-submodules https://github.com/Microsoft/cpprestsdk.git --depth 1
cd cpprestsdk
cmake .
make
make install

cd ..
rm -rf cpprestsdk

echo "Finished installing prereqs"
