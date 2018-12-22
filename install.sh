#!/bin/bash

echo "----- Install dependencies -----"
sudo apt-get update
sudo apt-get install cmake -y
sudo apt-get install dh-autoreconf -y

echo "----- M4ri -----"
git clone https://vkutuev@bitbucket.org/vkutuev/m4ri.git
cd m4ri/
autoreconf --install
./configure
sudo make install
cd ..

echo "----- Make program -----"
cmake . -DCMAKE_BUILD_TYPE=Release
make
