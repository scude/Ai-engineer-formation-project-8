# Documentation technique du projet de segmentation urbaine

## Préambule

Ce document récapitule l'intégralité des choix techniques réalisés pour la campagne Cityscapes conduite dans le notebook et la présentation finale. La structure d'origine est conservée : dataset et augmentations, panorama des modèles, focus DeepLabV3+, synthèse des résultats puis API Flask. L'objectif est de restituer l'essentiel en restant 40 % plus concis.

---

## 1. Dataset Cityscapes et stratégie d'augmentation

### 1.1. Présentation du dataset

Cityscapes fournit 5 000 scènes urbaines RGB (2 975 train / 500 val / 1 525 test) annotées pixel à pixel. Nous exploitons le remapping officiel « eight classes » regroupant 30 catégories en huit super-classes (route, trottoir, bâtiment, feu, végétation, ciel, personne, véhicule). Les masques conservent la valeur `255` comme `ignore_index`, écartée du calcul des pertes et métriques. Chaque image est ramenée à **512×1024** en PNG sans perte afin de limiter la VRAM tout en préservant les structures fines (piétons, feux).

### 1.2. Préparation et split des données

Le pipeline `build_dataset` (TensorFlow `tf.data`) prend en charge la lecture disque, le remapping et le calcul d'un masque de validité binaire synchronisé sur le masque remappé. Les données sont scindées en **80 % / 10 % / 10 %** pour entraîner, valider et réserver un lot test final. Chaque étape effectue : normalisation \[0 ; 1\], application des augmentations, redimensionnement différencié (bilinéaire pour l'image, *nearest* pour le masque) puis batching de deux échantillons avec prélecture (`prefetch`).

### 1.3. Choix d'Albumentations

Plusieurs augmentations géométriques (flips, crops, zooms, rotations) ont été testées dans le notebook. Elles n'ont apporté aucune hausse de `val_masked_mIoU` et allongeaient l'entraînement d'environ 12 % car DeepLabV3+ gère déjà ces invariances via son backbone ResNet50. Nous les avons donc retirées et conservé uniquement une corruption photométrique ou météo à la fois, orchestrée par Albumentations (`A.OneOf` parmi les quinze effets décrits par Bhuiya et al.).

Albumentations a été retenu face à `tf.image` ou imgaug car il est **simple d'usage**, **bien maintenu**, **populaire** et surtout parfaitement adapté à notre projet : propagation automatique des transformations sur les masques, définition déclarative claire et performances stables. L'API Flask réutilise la même pipeline pour garantir la cohérence entre entraînement, notebook et démonstration.

### 1.4. Impact mesuré des augmentations

Le protocole « une corruption à la fois » améliore en moyenne la `val_masked_mIoU` de +2,5 pts et réduit la variance inter-run. Les perturbations météo (brume, neige synthétique) et colorimétriques (CLAHE, ColorJitter) renforcent la robustesse sans dégrader les contours. Les distorsions optiques ont été plafonnées pour préserver les masques fins. Le notebook et la présentation indiquent la même conclusion : l'investissement temporel supplémentaire est acceptable au regard du gain de généralisation.

---

## 2. Modèles étudiés

### 2.1. Contexte expérimental commun

Tous les modèles ont été entraînés avec `SparseCategoricalCrossentropy` masquée, optimiseur SGD momentum 0,9 (scheduler polynomial, `poly_power=0,9`), et métriques `masked_mIoU`, `pixel_accuracy`, `dice_coef`. Les données proviennent du pipeline décrit ci-dessus, ce qui assure une comparaison équitable. Les logs et artefacts sont suivis dans MLflow, synchronisés avec la présentation finale.

### 2.2. Catalogue des architectures évaluées

#### U-Net mini

- **Rôle** : base légère pour vérifier le pipeline.
- **Architecture** : encodeur/décodeur symétriques à deux niveaux avec *skip connections* directes.
- **Résultats** : 0,32 de `val_masked_mIoU`, utile pour le débogage mais insuffisant pour la production.

#### U-Net VGG16

- **Backbone** : VGG16 pré-entraîné ImageNet, profondeur 23 couches.
- **Forces** : convergence rapide, segmentation visuellement fine.
- **Limites** : 14,7 M de paramètres, ~30 min/run, surapprentissage modéré (`val_masked_mIoU` ≈ 0,54).

#### MobileDet_seg

- **Inspiration** : MobileNetV2 avec blocs *inverted residual* et convolutions depthwise.
- **Décodeur** : trois *skip connections* et upsampling bilinéaire.
- **Résultats** : 0,50 de `val_masked_mIoU` pour 16 min d'entraînement, bon compromis embarqué.

#### YOLOv9_seg (simplifié)

- **Structure** : tronc CSP + tête PANet allégée convertie pour la segmentation dense.
- **Comportement** : rapide (10,5 min) mais perte d'information spatiale, `val_masked_mIoU` plafonné à 0,40.

#### DeepLabV3+ (ResNet50)

- **Composants** : backbone ResNet50 stride 16, tête ASPP (dilatations 1/6/12/18 + pooling global) et décodeur léger.
- **Atouts** : excellente séparation des classes, contours précis, meilleure stabilité train/val (`val_masked_mIoU` = 0,639).

#### Fast-SCNN (prototype)

- **Principe** : double branche détail/contexte avec mini-ASPP.
- **Statut** : essais 50 époques à 0,45 de `val_masked_mIoU`, non poursuivis faute de budget GPU.

---

## 3. Modèle retenu : DeepLabV3+ avec backbone ResNet50

### 3.1. Critères de sélection

DeepLabV3+ maximise la précision (`val_masked_mIoU`, `val_dice_coef`, `val_pix_acc`), présente un écart train/val maîtrisé et reste facilement industrialisable (export `.keras`, compatibilité TF-Lite/TensorRT). L'ASPP multi-échelle et le *skip* basse résolution expliquent la qualité des frontières (piétons, poteaux) tout en conservant le contexte global.

### 3.2. Hyperparamètres et pipeline d'entraînement

Les runs finaux durent jusqu'à 200 époques avec *early stopping* (patience 10). Le taux d'apprentissage initial est fixé à 1e-2 et décroît via le scheduler polynomial. Optuna (40 essais) a affiné les réglages : `lr≈8,5e-3`, `dice_loss_weight=0,15`, `aspp_dropout=0,1`, `poly_power=0,88`, ce qui apporte +1,8 pts de `val_masked_mIoU` par rapport aux hyperparamètres par défaut. Le pipeline Albumentations reste actif pendant l'entraînement final.

### 3.3. Artefacts générés et livraison

Deux artefacts principaux : `deeplab_resnet50_final.keras` (stocké dans `app/models/`) et les checkpoints nommés `{arch}.{monitor}.{epoch}-{score}.keras` archivés dans `artifacts/checkpoints/`. L'API Flask charge le modèle final au démarrage pour garantir que la meilleure itération est utilisée.

### 3.4. Analyse qualitative

Les masques produits conservent des bordures nettes route/trottoir, détectent véhicules et piétons malgré les occultations et segmentent mieux ciel/végétation que les modèles légers. Les erreurs restantes concernent les classes rares, mais elles sont moins fréquentes que sur U-Net VGG16 ou MobileDet.

### 3.5. Optimisation des hyperparamètres avec Optuna

Le notebook utilise Optuna (`TPESampler`) pour piloter `TrainConfig`. L'espace de recherche couvre le taux d'apprentissage (5e-4→2e-2), le poids Dice (0→0,5), le dropout ASPP (0→0,3) et `poly_power` (0,7→1,0). Chaque essai s'arrête à 35 époques via *early stopping*, soit ~6 min sur GPU T4. Les meilleurs paramètres sont journalisés (MLflow + `study.best_params`) et réinjectés automatiquement, assurant la reproductibilité signalée dans la présentation.

---

## 4. Benchmarking des résultats

### 4.1. Synthèse des métriques

Les métriques ci-dessous proviennent directement du notebook (section « Faire le point sur les résultats intermédiaires ») et sont reprises telles quelles dans la présentation finale.

| Modèle                     | Durée (min) | `masked_mIoU` (train) | `val_masked_mIoU` | `pix_acc` (train) | `val_pix_acc` | `dice_coef` (train) | `val_dice_coef` |
| :------------------------- | :---------: | :-------------------: | :---------------: | :---------------: | :-----------: | :-----------------: | :-------------: |
| **DeepLabV3+ (ResNet50)**  | **13,4**    | **0,947**             | **0,639**         | **0,989**         | **0,872**     | **0,965**           | **0,716**       |
| **YOLOv9_seg (simplifié)** | 10,5        | 0,689                 | 0,400             | 0,913             | 0,714         | 0,753               | 0,494           |
| **MobileDet_seg**          | 16,3        | 0,938                 | 0,502             | 0,987             | 0,779         | 0,953               | 0,600           |
| **U-Net VGG16**            | 29,7        | 0,903                 | 0,542             | 0,977             | 0,805         | 0,923               | 0,633           |
| **U-Net mini**             | 6,1         | 0,563                 | 0,319             | 0,851             | 0,634         | 0,650               | 0,407           |

### 4.2. Interprétation des indicateurs

- **Masked mIoU** : DeepLabV3+ domine (0,639), U-Net VGG16 et MobileDet forment le second groupe (0,54 et 0,50). Les architectures plus légères décrochent sous 0,40 faute de contexte.
- **Pixel accuracy** : DeepLab atteint 0,872, U-Net VGG16 reste à 0,805, MobileDet à 0,779. U-Net mini et YOLOv9 simplifié peinent à franchir 0,72.
- **Dice coefficient** : l'écart train/val reste contenu pour DeepLab (0,965→0,716), plus marqué pour U-Net VGG16 (0,923→0,633) et MobileDet (0,953→0,600), signe d'une régularisation plus fragile.

### 4.3. Influence des augmentations sur DeepLabV3+

| Configuration                              | Durée d'entraînement | `val_dice_coef` | `val_masked_mIoU` | `val_pix_acc` | Conclusions |
| :----------------------------------------- | :------------------: | :-------------: | :----------------: | :------------: | :--------- |
| DeepLabV3+ (sans augmentation)             | 3,9 h               | 0,840           | 0,818              | 0,945          | Peu d'erreurs globales mais bords plus flous. |
| DeepLabV3+ (corruption photométrique unique)| 6,7 h               | **0,849**       | **0,831**          | **0,948**      | Masques plus nets et robustes malgré +3 h. |

Ces chiffres confirment que les variations photométriques/météo apportent un gain léger mais constant, tandis que les essais géométriques étaient superflus.

### 4.4. Analyse multi-critères

| Critère                              | Modèle recommandé             | Rationale |
| :----------------------------------- | :---------------------------- | :-------- |
| Précision globale (mIoU / Dice)      | **DeepLabV3+ ResNet50**       | Meilleure performance et stabilité. |
| Compromis vitesse / qualité          | **MobileDet_seg**             | Entraînement modéré, inférence légère. |
| Haute fidélité visuelle (VRAM ample) | **U-Net VGG16**               | Rendu très détaillé mais coûteux. |
| Prototypage / tests pipeline         | **U-Net mini**                | Rapide à entraîner, utile pour valider les scripts. |

### 4.5. Observations complémentaires

- Les corruptions Albumentations renforcent la généralisation de DeepLab et U-Net VGG16 tout en restant neutres sur MobileDet.
- YOLOv9 simplifié souffre de l'absence de *skip connections* riches ; ses masques restent bruités sur trottoirs et bâtiments.
- Fast-SCNN, bien que prometteur, nécessite plus d'époques pour exprimer son potentiel et figure dans les travaux futurs.

---

## 5. API Flask, architecture d'inférence et conclusion ouverte

### 5.1. Vue d'ensemble applicative

`run.py` instancie Flask et enregistre deux services : `SegmentationService` (chargement du modèle, pré/post-traitements, prédiction) et `AugmentationService` (tirage aléatoire ou galerie complète des 15 corruptions). La même pipeline Albumentations que celle d'entraînement est utilisée pour garantir la cohérence des démonstrations.

### 5.2. SegmentationService

1. Chargement du modèle `.keras` au démarrage.
2. Redimensionnement bilinéaire vers 512×1024, normalisation \[0 ; 1\], ajout de la dimension batch.
3. Prédiction `model.predict`, `argmax` sur l'axe classe.
4. Redimensionnement inverse du masque (*nearest neighbor*), colorisation via `PALETTE`, fusion alpha 0,5 avec l'image d'origine.
5. Sérialisation en data URLs via `SegmentationResult` pour l'API JSON.

### 5.3. AugmentationService

- `generate` renvoie l'image originale et `samples` corruptions aléatoires issues du `A.OneOf` (identique à l'entraînement).
- `gallery` applique séquentiellement les 15 transformations pour inspection détaillée.
- Les sorties sont emballées dans `AugmentedImage` puis converties en data URLs côté route.

### 5.4. Endpoints REST

- **`/predict`** : fichier `image` requis, renvoie `original`, `mask`, `overlay`.
- **`/augment`** : retourne des corruptions aléatoires (paramètre `samples`).
- **`/augment/gallery`** : livre l'ensemble des effets, utile pour la présentation.

Les payloads sont limités à 16 MiB (`MAX_CONTENT_LENGTH`). En absence de fichier, l'API renvoie un 400 explicite.

### 5.5. Déploiement, dépendances et hébergement Heroku

- **Dépendances** : Python 3.10+, TensorFlow 2.12, Albumentations, Flask, Pillow, numpy (voir `requirements.txt`).
- **Docker** : image de base `python:3.10-slim`, lancement via `docker build -t cityscapes-seg .` puis `docker run -p 5000:5000 cityscapes-seg`.
- **Heroku Container** : `heroku login`, `heroku create`, `heroku stack:set container`, `heroku container:push web`, `heroku container:release web`, `heroku logs --tail` pour la vérification. Variable recommandée : `MODEL_PATH=app/models/deeplab_resnet50_final.keras`.
- **Scalabilité** : possibilité de `heroku ps:scale web=2` et d'activer l'auto-scaling. Pour plus de throughput, envisager TensorRT ou une déclinaison TFLite quantifiée.

### 5.6. Conclusion ouverte et pistes d'évolution

1. **Optimisation temps réel** : conversion TensorRT/TFLite pour accélérer l'inférence sur GPU ou CPU embarqué.
2. **Benchmark élargi** : intégrer Fast-SCNN, BiSeNet, DDRNet afin d'explorer davantage le compromis précision/latence.
3. **Extension datasets** : croiser Cityscapes avec Mapillary ou BDD100K pour enrichir les classes.
4. **Monitoring** : ajouter endpoints de santé, métriques Prometheus et journaux centralisés.
5. **Expérience utilisateur** : contrôler l'opacité du masque, exporter en GeoJSON, proposer un mode batch ou streaming.

---

En synthèse, la combinaison DeepLabV3+ + Albumentations photométriques offre le meilleur compromis précision/robustesse pour notre segmentation urbaine huit classes. L'API Flask met directement à disposition ce pipeline reproductible pour la démonstration comme pour un déploiement rapide.
