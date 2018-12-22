#!/bin/bash

echo "----- Install dependencies -----"
apt-get update
apt-get install -y cmake

git clone https://vkutuev@bitbucket.org/vkutuev/m4ri.git
cd m4ri/
autoreconf --install
./configure
make install
cd ..

cmake . -DCMAKE_BUILD_TYPE=Release -j 4
make
