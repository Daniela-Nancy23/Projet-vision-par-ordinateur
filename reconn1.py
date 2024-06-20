import cv2
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image

# Charger votre modèle
model= tf.keras.models.load_model('essa/model.keras')

# Ouvrir la webcam
cap = cv2.VideoCapture(0)


# Liste pour garder une trace des 5 dernières prédictions
last_five_predictions = []

while(True):
    # Capturer une image de la webcam
    ret, frame = cap.read()

    # Prétraiter l'image si nécessaire (selon votre modèle)
    if frame is not None:
       cv2.imshow('frame', frame)
    if len(frame.shape) == 2:
       frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
       cv2.imshow('frame', frame)
    if frame.shape[2] > 3:
       frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
       cv2.imshow('frame', frame)

    # Redimensionner l'image
    frame = cv2.resize(frame, (150, 150))

    # Ajouter une dimension supplémentaire pour indiquer le nombre d'échantillons
    frame = np.expand_dims(frame, axis=0)

    # Prédire le niveau d'intérêt
    prediction = model.predict(frame)

    # Afficher la prédiction
    print(prediction)
    predicted_class = np.argmax(prediction)

    # Afficher la classe prédite
    class_names = [ 'beaucoup interesse', 'pas interesse', 'peu interesse']
    print('Classe prédite :', class_names[predicted_class])

    # Ajouter la prédiction à la liste des 5 dernières prédictions
    last_five_predictions.append(class_names[predicted_class])

    # Vérifier si la liste a atteint 5 éléments
    if len(last_five_predictions) == 5:
        # Vérifier si tous les éléments sont "beaucoup interesse"
        if all(pred == 'beaucoup interesse' for pred in last_five_predictions):
            from tkinter import Tk, Canvas, Label, font
            from PIL import ImageTk, Image

            def clignoter(label):#Cette fonction fait clignoter le texte du label en changeant sa couleur
             current_color = label.cget("foreground")
             next_color = "red" if current_color == couleur_texte else couleur_texte
             label.config(foreground=next_color)
             root.after(500, clignoter, label)  # Continuez à appeler cette fonction toutes les 500 millisecondes

            # Créez la fenêtre root
            root = Tk()
            root.geometry('600x800')  # Augmentez la taille de la fenêtre pour inclure des marges
            root.configure(bg='white')  # Fond blanc pour l'ensemble de l'interface

            # Définissez la police de caractères
            police_grande = font.Font(family='Helvetica', size=20, weight='bold')
            police_grande1 = font.Font(family='Times', size=20, slant='italic', weight='bold')
            couleur_texte = "#3d3d3d"  # Plus doux pour les yeux

            # Chargez l'image
            image = Image.open('C:/Users/Utilisateur/Downloads/shop.jpg')
            image = image.resize((500, 500))  # Redimensionnez l'image
            photo = ImageTk.PhotoImage(image)

            # Créez la toile
            canvas = Canvas(root, width=500, height=500, bg='white', bd=0, highlightthickness=0)
            canvas.pack(side='top', pady=20)  # Ajoutez un padding pour créer des marges

            # Dessinez l'image sur la toile, centrée
            canvas.create_image(250, 250, image=photo)

            # Ajoutez le texte sur le canvas
            canvas.create_text(250, 250, text="Bienvenue chez\nFreeShop", font=police_grande1, fill=couleur_texte,
                               justify='center')

            # Créez les labels pour le texte suivant
            lab4 = Label(root, text='Un client détecté!', font=police_grande, bg='white', fg=couleur_texte)
            lab4.pack(pady=10)  # Diviseur vertical pour séparer chaque information
            lab1 = Label(root, text='Beaucoup interessé', font=police_grande, bg='white', fg=couleur_texte)
            lab1.pack(pady=10)
            lab2 = Label(root, text='Rapprochez-vous de ce client!', font=police_grande, bg='white', fg=couleur_texte)
            lab2.pack(pady=10)
            # Fermez la fenêtre après un délai (5000 millisecondes = 5 secondes)
            root.after(5000, root.destroy)

            clignoter(lab1)
            root.mainloop()
        if all(pred == 'peu interesse' for pred in last_five_predictions):
            from tkinter import Tk, Canvas, Label, font
            from PIL import ImageTk, Image

            def clignoter(label):#Cette fonction fait clignoter le texte du label en changeant sa couleur
             current_color = label.cget("foreground")
             next_color = "yellow" if current_color == couleur_texte else couleur_texte
             label.config(foreground=next_color)
             root.after(500, clignoter, label)  # Continuez à appeler cette f
            # Créez la fenêtre root
            root = Tk()
            root.geometry('600x800')  # Augmentez la taille de la fenêtre pour inclure des marges
            root.configure(bg='white')  # Fond blanc pour l'ensemble de l'interface

            # Définissez la police de caractères
            police_grande = font.Font(family='Helvetica', size=20, weight='bold')
            police_grande1 = font.Font(family='Times', size=20, slant='italic', weight='bold')
            couleur_texte = "#3d3d3d"  # Plus doux pour les yeux

            # Chargez l'image
            image = Image.open('C:/Users/Utilisateur/Downloads/shop.jpg')
            image = image.resize((500, 500))  # Redimensionnez l'image
            photo = ImageTk.PhotoImage(image)

            # Créez la toile
            canvas = Canvas(root, width=500, height=500, bg='white', bd=0, highlightthickness=0)
            canvas.pack(side='top', pady=20)  # Ajoutez un padding pour créer des marges

            # Dessinez l'image sur la toile, centrée
            canvas.create_image(250, 250, image=photo)

            # Ajoutez le texte sur le canvas
            canvas.create_text(250, 250, text="Bienvenue chez\nFreeShop", font=police_grande1, fill=couleur_texte,
                               justify='center')

            # Créez les labels pour le texte suivant
            lab4 = Label(root, text='Un client détecté!', font=police_grande, bg='white', fg=couleur_texte)
            lab4.pack(pady=10)  # Diviseur vertical pour séparer chaque information
            lab1 = Label(root, text='peu interessé', font=police_grande, bg='white', fg=couleur_texte)
            lab1.pack(pady=10)
            lab2 = Label(root, text='Rapprochez-vous de ce client!', font=police_grande, bg='white', fg=couleur_texte)
            lab2.pack(pady=10)
            # Fermez la fenêtre après un délai (5000 millisecondes = 5 secondes)
            root.after(5000, root.destroy)

            clignoter(lab1)
            root.mainloop()

        # Effacer la liste
        last_five_predictions = []

    # Ajoutez un délai avant de passer à la frame suivante
    time.sleep(2)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer la webcam
cap.release()
cv2.destroyAllWindows()
