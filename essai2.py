import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image



# Répertoires des vidéos
base_dir = 'essa'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_beaucoup_interesse_dir = os.path.join(train_dir, 'beaucoup interesse')
train_pas_interesse_dir = os.path.join(train_dir, 'pas interesse')
train_peu_interesse_dir = os.path.join(train_dir, 'peu interesse')

validation_beaucoup_interesse_dir = os.path.join(validation_dir, 'beaucoup interesse')
validation_pas_interesse_dir = os.path.join(validation_dir, 'pas interesse')
validation_peu_interesse_dir = os.path.join(validation_dir, 'peu interesse')

# Dimensions de l'image
IMG_HEIGHT = 150
IMG_WIDTH = 150
NUM_CLASSES = 3

# Chemin vers votre vidéo
video_path = ('essa/train/pas interesse/WhatsApp Video 2024-06-13 at 13.38.18(2).mp4')
# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)


def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frames.append(frame)
    cap.release()
    return frames

def load_data_from_videos(video_dirs, max_frames_per_video=30):
    X, y = [], []
    for label, video_dir in enumerate(video_dirs):
        for video_file in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_file)
            frames = extract_frames(video_path, max_frames=max_frames_per_video)
            X.extend(frames)
            y.extend([label] * len(frames))
    return np.array(X), np.array(y)

# Créer un dossier pour stocker les images extraites
os.makedirs('images9', exist_ok=True)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionner l'image
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    # Sauvegarder l'image
    img_path = f'images9/frame_{frame_count}.jpg'
    cv2.imwrite(img_path, frame)

    frame_count += 1

cap.release()


# Chargement des données d'entraînement et de validation
train_dirs = [train_beaucoup_interesse_dir, train_pas_interesse_dir, train_peu_interesse_dir]
validation_dirs = [ validation_beaucoup_interesse_dir, validation_pas_interesse_dir, validation_peu_interesse_dir]

X_train, y_train = load_data_from_videos(train_dirs)
X_val, y_val = load_data_from_videos(validation_dirs)

# Normalisation des images
X_train = X_train / 255.0
X_val = X_val / 255.0

# Création du modèle
model = models.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Évaluation du modèle
loss, accuracy = model.evaluate(X_val,y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Sauvegarder le modèle
model.save('essa/model.keras')

# Charger l'image
img_path = 'images9/frame_0.jpg'
img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

# Convertir l'image en tableau numpy et ajouter une dimension supplémentaire
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Normaliser l'image
x = x / 255.0

# Faire la prédiction
predictions = model.predict(x)

# Obtenir l'indice de la classe prédite
predicted_class = np.argmax(predictions)

# Afficher la classe prédite
class_names = ['beaucoup interesse',  'pas interesse', 'peu interesse']
print('Classe prédite :', class_names[predicted_class])

