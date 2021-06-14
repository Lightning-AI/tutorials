#!/bin/bash

printf "Datasets: $1\n"

mkdir -p $1
cd $1

wget www.di.ens.fr/~lelarge/MNIST.tar.gz --progress=bar:force:noscroll
tar -zxvf MNIST.tar.gz
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz --progress=bar:force:noscroll
tar -zxvf cifar-10-python.tar.gz

echo $PWD
ls -l
