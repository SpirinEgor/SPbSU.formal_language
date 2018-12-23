#!/bin/bash

echo "----- Install dependencies -----"
sudo apt-get update
sudo apt-get install -y dh-autoreconf wget

echo "----- CMake -----"
wget https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2-Linux-x86_64.tar.gz
tar xzf cmake-3.13.2-Linux-x86_64.tar.gz
rm cmake-3.13.2-Linux-x86_64.tar.gz

echo "----- M4ri -----"
git clone https://vkutuev@bitbucket.org/vkutuev/m4ri.git
cd m4ri/
autoreconf --install
autoreconf --install
./configure
sudo make install
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib"' >> $BASH_ENV
cd ..

echo "----- Make program -----"
cmake-3.13.2-Linux-x86_64/bin/cmake . -DCMAKE_BUILD_TYPE=Release
make
