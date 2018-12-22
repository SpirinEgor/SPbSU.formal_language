#!/bin/bash

echo "----- Install dependencies -----"
# sudo apt-get update
sudo apt-get install cmake
sudo apt-get install autoconf

# wget http://ftp.gnu.org/gnu/autoconf/autoconf-2.69.tar.gz 
# tar xf autoconf*
# cd autoconf-2.69
# sh configure --prefix /usr/local
# make install

# wget http://ftp.gnu.org/gnu/automake/automake-1.15.tar.gz
# tar xf automake*
# cd automake-1.15
# sh configure --prefix /usr/local
# make install

# wget http://mirror.jre655.com/GNU/libtool/libtool-2.4.6.tar.gz
# tar xf libtool*
# cd libtool-2.4.6
# sh configure --prefix /usr/local
# make install 

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
