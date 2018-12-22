#!/bin/bash

echo "----- Install dependencies -----"
sudo apt-get update
sudo apt-get install cmake autoconf

git clone https://vkutuev@bitbucket.org/vkutuev/m4ri.git
cd m4ri/
autoreconf --install
./configure
sudo make install
cd ..

cmake . -DCMAKE_BUILD_TYPE=Release -j 4
make
