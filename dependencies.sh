#!/bin/bash

echo "Mise à jour des paquets..."
sudo apt-get update

echo "Installation de Python3-pip..."
sudo apt-get install -y python3-pip

# Installation de Tkinter pour Python3
echo "Installation de Tkinter..."
sudo apt-get install -y python3-tk

# Mise à niveau de pip à la dernière version
echo "Mise à niveau de pip..."
python3 -m pip install --upgrade pip

# Installation des bibliothèques Python nécessaires
echo "Installation de TensorFlow, PyTorch, OpenCV-Python et Pillow..."
python3 -m pip install  torch opencv-python pillow
#pip install tensorflow
echo "Toutes les dépendances ont été installées avec succès."
