#!/usr/bin/env bash
# Install jupyter

cd ~/
wet https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
sh Anaconda3-2019.10-Linux-x86_64.sh

source ~/.bashrc


# config jupyter
jupyter notebook --generate-config

vi ~/.jupyter/jupyter_notebook_config.py

c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 5000

# run jupyter
jupyter-notebook --no-browser --port=5000

# su vi /etc/rc.local
# su zooicl84 -c "
#
#!/bin/bash
jupyter-notebook --config=/home/zooicl84/.jupyter/jupyter_notebook_config.py --no-browser --port=5000


# CMake
sudo apt instasll cmake

sudo -E add-apt-repository -y ppa:george-edison55/cmake-3.x
sudo -E apt-get update
sudo apt-get install cmake

sudo apt install g++

sudo apt-get update
sudo apt-get install --no-install-recommends nvidia-375
sudo apt-get install --no-install-recommends nvidia-opencl-icd-375 nvidia-opencl-dev opencl-headers


# Lightgbm
cd ~/
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake -DUSE_GPU=1 ..
make -j16

sudo apt-get -y install python-pip
sudo -H pip install setuptools numpy scipy scikit-learn -U
cd ..
cd ~/
cd LightGBM/python-torch_design
sudo python setup.py install --precompile




# Mecab
cd ~/
git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
bash Mecab-ko-for-Google-Colab/install_mecab-ko_on_colab190912.sh