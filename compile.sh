#!/bin/bash

g++ ./src/winter.cc -I ./src/headers/ -o winter -std=c++11
./winter;
rm winter;
