#!/bin/bash

echo "----- Install dependencies -----"
sudo apt-get update
sudo apt-get install dh-autoreconf wget -y

echo "----- CMake -----"
wget https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.tar.gz
# tar xzf cmake-3.13.2-Linux-x86_64.tar.gz
# cd cmake-3.13.2-Linux-x86_64

echo "----- M4ri -----"
git clone https://vkutuev@bitbucket.org/vkutuev/m4ri.git
cd m4ri/
autoreconf --install
./configure
sudo make install
cd ..

echo "----- Make program -----"
/cmake-3.13.2-Linux-x86_64/bin/cmake . -DCMAKE_BUILD_TYPE=Release
make
