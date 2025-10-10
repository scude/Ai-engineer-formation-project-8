# Guide de démarrage de l'application Flask

Ce projet fournit une interface web simple pour tester un modèle de segmentation sémantique et visualiser le pipeline d'augmentation utilisé pendant l'entraînement. Le serveur expose à la fois une UI Bootstrap et une API REST (`/predict`, `/augment`).

## Prérequis
- Python 3.10 ou supérieur
- `pip` et `virtualenv`
- Le fichier du modèle TensorFlow entraîné : `deeplab_resnet50_final.keras`

## Installation
1. Cloner le dépôt et se placer à sa racine :
   ```bash
   git clone <URL_DU_DEPOT>
   cd Ai-engineer-formation-project-8
   ```
2. Créer un environnement virtuel puis installer les dépendances Python :
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sous Windows : .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Copier le modèle entraîné dans `app/models/` :
   ```bash
   cp notebook/artifacts/deeplab_resnet50_final.keras app/models/
   ```
   > Si le fichier provient d'une autre machine, placez-le dans ce dossier avec exactement le même nom.

## Lancement du serveur
1. Exporter les variables d'environnement nécessaires si vous souhaitez changer la configuration (optionnel). Par défaut, l'application utilise `run.py` pour créer l'app Flask.
2. Démarrer le serveur de développement :
   ```bash
   python run.py
   ```
3. Ouvrir un navigateur et visiter http://127.0.0.1:5000/.

## Fonctionnalités
- **Onglet "Segmentation"** : téléchargez une image pour afficher l'original, le masque colorisé et la superposition avec la légende des classes.
- **Onglet "Augmentation"** : téléchargez une image pour visualiser un échantillon d'augmentations géométriques et photométriques (flip, rotation, recadrage, luminosité, contraste, etc.).
- Les mêmes transformations que pendant l'entraînement sont appliquées de manière déterministe côté UI afin de reproduire fidèlement le pipeline.

## API REST
Vous pouvez appeler directement les endpoints JSON :
- `POST /predict` avec un fichier `image` multipart pour obtenir la prédiction et les images encodées en base64.
- `POST /augment` avec un fichier `image` pour récupérer un lot d'exemples augmentés.

## Dépannage
- Vérifiez que le modèle `app/models/deeplab_resnet50_final.keras` est présent avant de démarrer, sinon l'application échouera au chargement.
- Le poids maximal accepté pour un fichier est de 16 MiB (`MAX_CONTENT_LENGTH`).

## Tests rapides
Pour vérifier que tout le code se compile correctement :
```bash
python -m compileall app
```
heroku container:push web -a city-segmentation
heroku container:release web -a city-segmentation
heroku logs -t -a city-segmentation

Bonne exploration !