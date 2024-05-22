import tkinter as tk
from tkinter import filedialog, Radiobutton, IntVar, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np


# Fonctions pour charger les modèles
def load_sift_model():
    sift = cv2.SIFT_create()
    return sift

def load_fast_model():
    fast = cv2.FastFeatureDetector_create()
    return fast

def load_censure_model():
    star = cv2.xfeatures2d.StarDetector_create()
    return star
def load_brief_model():
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    return brief

# Fonctions de prédiction pour chaque modèle
def predict_with_sift(model, image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, _ = model.detectAndCompute(gray, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_with_keypoints, len(keypoints)

def predict_with_fast(model, image_path):
    img = cv2.imread(image_path, 0)
    keypoints = model.detect(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_with_keypoints, len(keypoints)

def predict_with_censure(model, image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = model.detect(gray, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_with_keypoints, len(keypoints)



def predict_with_brief(detector_model, descriptor_model, image_path):
    img = cv2.imread(image_path, 0)  # Charger l'image en niveaux de gris
    keypoints = detector_model.detect(img, None)  # Détecter les points clés avec FAST ou tout autre détecteur
    keypoints, descriptors = descriptor_model.compute(img, keypoints)  # Calculer les descripteurs avec BRIEF
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_with_keypoints, len(keypoints)

# Variable globale pour gérer l'image originale et le zoom
original_img = None
zoom_level = 1

# Fonction pour sauvegarder l'image
def save_image():
    global processed_img_cv
    if processed_img_cv is not None:
        filepath = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
        if filepath:
            cv2.imwrite(filepath, processed_img_cv)
            messagebox.showinfo("Sauvegarde", "L'image a été sauvegardée avec succès.")
# Fonction pour ajuster le zoom
def adjust_image_zoom(zoom_factor, panel):
    global original_img, zoom_level
    if original_img:
        zoom_level *= zoom_factor
        width, height = original_img.size
        new_size = int(width * zoom_level), int(height * zoom_level)
        img_resized = original_img.resize(new_size, Image.Resampling.LANCZOS)
        img_display = ImageTk.PhotoImage(img_resized)
        panel.configure(image=img_display)
        panel.image = img_display

# Fonction principale pour ouvrir une image et prédire
def open_image_and_predict(model_choice, panel, result_label):
    file_path = filedialog.askopenfilename()
    if file_path:
        
        global original_img, zoom_level
        zoom_level = 1  # Reset zoom level
        if model_choice.get() == 5:  # FAST sélectionné
            fast_model = load_fast_model()
            img_with_keypoints, num_keypoints = predict_with_fast(fast_model, file_path)
            img_with_keypoints = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_with_keypoints)
            result_label.config(text=f"Points clés détectés avec FAST : {num_keypoints}")
        elif model_choice.get() == 4:  # SIFT sélectionné
            sift_model = load_sift_model()
            img_with_keypoints, num_keypoints = predict_with_sift(sift_model, file_path)
            img_with_keypoints = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_with_keypoints)
            result_label.config(text=f"Points clés détectés avec SIFT : {num_keypoints}")
        elif model_choice.get() == 6:  # CenSurE sélectionné
            censure_model = load_censure_model()
            img_with_keypoints, num_keypoints = predict_with_censure(censure_model, file_path)
            img_with_keypoints = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_with_keypoints)
            result_label.config(text=f"Points clés détectés avec CenSurE : {num_keypoints}")
            
        elif model_choice.get() == 7:  # BRIEF sélectionné
            fast_model = load_fast_model()
            brief_model = load_brief_model()
            img_with_keypoints, num_keypoints = predict_with_brief(fast_model, brief_model, file_path)
            img_with_keypoints = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_with_keypoints)
            result_label.config(text=f"Points clés détectés avec BRIEF : {num_keypoints}")
        else:
            # Ajouter ici la logique pour SIFT, FAST, et autres modèles si nécessaire
            pil_img = Image.open(file_path)
            result_label.config(text="Image chargée")

        original_img = pil_img
        adjust_image_zoom(1, panel)  # Afficher l'image initiale sans zoom

# Initialisation de l'interface utilisateur Tkinter
root = tk.Tk()
root.title("IA Image Predictor avec Zoom, SIFT, FAST et CenSurE")

model_choice = IntVar(value=1)

# Interface pour le choix du modèle
Radiobutton(root, text="SIFT", variable=model_choice, value=4).pack(anchor=tk.W)
Radiobutton(root, text="FAST", variable=model_choice, value=5).pack(anchor=tk.W)
Radiobutton(root, text="CenSurE", variable=model_choice, value=6).pack(anchor=tk.W)  # Ajout de CenSurE
Radiobutton(root, text="BRIEF", variable=model_choice, value=7).pack(anchor=tk.W)

open_button = Button(root, text="Ouvrir une image et prédire", command=lambda: open_image_and_predict(model_choice, panel, result_label))
open_button.pack()

# Boutons de zoom
zoom_in_button = Button(root, text="Zoom +", command=lambda: adjust_image_zoom(1.25, panel))
zoom_in_button.pack(side=tk.LEFT)

zoom_out_button = Button(root, text="Zoom -", command=lambda: adjust_image_zoom(0.8, panel))
zoom_out_button.pack(side=tk.RIGHT)

panel = Label(root)
panel.pack()

result_label = Label(root, text="Résultat de prédiction s'affichera ici")
result_label.pack()

root.mainloop()
