
Write-Host "Mise à niveau de pip à la dernière version..."
python -m pip install --upgrade pip


Write-Host "Installation de TensorFlow..."
#python -m pip install tensorflow==2.4.0

# Installation de PyTorch

Write-Host "Installation de PyTorch..."
python -m pip install torch torchvision torchaudio

# Installation d'OpenCV
Write-Host "Installation d'OpenCV-Python..."

python -m pip install opencv-python
Write-Host "Installation de opencv-contrib-python "
pip install opencv-contrib-python
# Installation de Pillow
Write-Host "Installation de Pillow..."
python -m pip install Pillow

Write-Host "Toutes les dépendances ont été installées avec succès."
