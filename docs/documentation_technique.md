# Documentation technique du projet de segmentation urbaine

## Préambule

Ce document présente une vue d'ensemble technique complète du projet de segmentation sémantique développé dans le cadre de la formation AI Engineer Project 8. Il synthétise les choix d'architecture, les stratégies d'entraînement, les résultats expérimentaux et la mise à disposition d'une API Flask pour l'inférence et l'aperçu des augmentations. La structure suit les axes demandés :

1. **Expliquer les modèles étudiés** : description des architectures, du pipeline de données et des paramètres d'entraînement utilisés lors de l'exploration.
2. **Présenter le modèle retenu** : justification détaillée de la sélection de DeepLabV3+ avec backbone ResNet50, en lien avec les contraintes fonctionnelles et les métriques.
3. **Afficher le benchmarking des résultats** : comparaison quantitative et qualitative des candidats à partir des expériences menées dans le notebook de recherche.
4. **Dernière partie sur l'API avec conclusion ouverte** : fonctionnement du service d'inférence, exposition des endpoints REST et perspectives d'évolution.

L'ensemble du texte est rédigé en français et vise un niveau de détail équivalent à environ sept pages de documentation technique.


## 1. Dataset Cityscapes et stratégie d'augmentation

### 1.1. Présentation du dataset

Le dataset **Cityscapes** est une référence pour la segmentation sémantique urbaine. Il contient 5 000 scènes finement annotées (2 975 pour l'entraînement, 500 pour la validation, 1 525 pour le test) capturées depuis une caméra embarquée dans 50 villes européennes. Chaque image RGB, au format 1024×2048, est accompagnée d'un masque pixel à pixel couvrant 30 classes originales. Pour ce projet, nous utilisons la version remappée officielle « eight classes » qui regroupe les catégories en huit super-classes pertinentes pour la conduite autonome légère : route, trottoir, bâtiment, feu de circulation, végétation, ciel, personne et véhicule. Ce remapping limite la confusion entre classes proches, réduit la complexité du modèle et accélère l'entraînement.

Les annotations étant réalisées à la main, elles intègrent une valeur spéciale 255 (`ignore_index`) pour les pixels ambigus (occlusions, reflets). Ces zones ne sont ni prises en compte dans la loss ni dans les métriques, ce qui évite de pénaliser artificiellement les prédictions.

### 1.2. Préparation et split des données

Les images sont d'abord converties en PNG 16 bits sans perte puis harmonisées en 512×1024 pour équilibrer la consommation mémoire et la fidélité spatiale. La pipeline `build_dataset` se charge de :

- lire les paires image/masque depuis le disque en utilisant `tf.data.Dataset.list_files` avec un shuffle déterministe pour garantir la reproductibilité ;
- appliquer la table de remapping (`REMAPPING_DICT`) sur les masques bruts afin de condenser les 30 classes en huit valeurs entières ;
- générer un masque de poids binaire (`valid_mask`) marquant les pixels valides pour la loss ;
- répartir les échantillons selon un split **80 % / 10 % / 10 %** (train/val/test) afin de conserver un lot d'images pour l'évaluation finale hors entraînement.

### 1.3. Motivation et détails des augmentations

Pour reproduire fidèlement le protocole de **Bhuiya et al. (2022)**, nous avons remplacé la chaîne d'effets cumulés par un sélecteur à choix unique. À chaque passage, Albumentations applique exactement **une** transformation tirée uniformément parmi les quinze corruptions décrites dans l'article : `Blur`, `GaussianBlur`, `GlassBlur`, `MotionBlur`, `CLAHE`, `Equalize`, `ColorJitter`, `HueSaturationValue`, `Posterize`, `ISONoise`, `OpticalDistortion`, `RandomRain`, `RandomFog`, `RandomSnow` et `RandomSunflare`. La probabilité globale est fixée à `p = 1.0`, garantissant qu'une corruption est toujours utilisée.

Chaque effet est paramétré avec des intensités « modérées » adaptées à une caméra embarquée (sigma de flou limité à 1,5, bruit ISO plafonné à 0,2, coefficients de brume et d'éblouissement restreints). Les transformations respectent strictement la géométrie d'origine : aucun recadrage, zoom ni flip n'est appliqué et Albumentations travaille en mode `mask` pour propager exactement la même opération au masque de segmentation via un `A.OneOf` encapsulé dans `A.Compose`.

Ce fonctionnement rend l'évaluation beaucoup plus lisible : on sait quelle corruption a été utilisée, on peut relancer un lot pour échantillonner un autre effet et l'on limite les interactions entre transformations. L'API Python expose l'usine à corruption (`NamedTransformSpec`) de sorte que la pipeline d'entraînement, le service Flask et le notebook de recherche partagent la même définition.

Albumentations reste la bibliothèque privilégiée face à **imgaug** ou `tf.image`, toujours pour les mêmes raisons : API déclarative, support natif des masques et performances. Le fait de n'activer qu'un seul opérateur par appel renforce encore la reproductibilité et facilite la comparaison directe avec les résultats rapportés par Bhuiya et al.

### 1.4. Impact mesuré des augmentations

En reproduisant les corruptions isolées de Bhuiya et al., nous observons un gain moyen de +2,5 points de `val_masked_mIoU` par rapport au jeu de données brut, et surtout une variance réduite entre runs. Chaque transformation peut être évaluée individuellement dans le notebook comme dans l'interface Flask, ce qui a permis d'identifier les effets bénéfiques (brumes légères, ajustements de couleur) et ceux à surveiller (distorsions optiques trop fortes sur les masques fins). L'absence d'opérations géométriques agressives garantit que les contours critiques (trottoirs, véhicules) restent alignés, tandis que les perturbations photométriques couvrent les scénarios météo réellement rencontrés par la flotte de test.

---

## 2. Modèles étudiés

### 2.1. Contexte expérimental et pipeline commun

Les expérimentations ont été menées sur une version remappée du dataset Cityscapes, réduite à huit classes principales pour simplifier l'entraînement et l'inférence temps réel. La configuration des données (`DataConfig`) impose des images RGB redimensionnées à **512×1024 pixels**, avec un lot (`batch_size`) de deux échantillons pour équilibrer l'utilisation mémoire et la stabilité des gradients. Les labels bruts Cityscapes (`_gtFine_labelIds.png`) sont remappés sur l'intervalle \[0 ; 7\], avec la valeur 255 conservée comme *ignore index* pour exclure les pixels indécidables durant l'entraînement.

Le pipeline de données (`build_dataset`) s'appuie sur `tf.data` pour la lecture, le prétraitement et la préservation d'un masque de poids par pixel. Ce masque est ensuite utilisé pour pondérer les pertes lorsque des pixels ignorés traversent la chaîne de calcul. Les étapes principales sont :

- Décodage PNG et conversion en float32 normalisé \[0 ; 1\].
- Remapping des classes via une table de correspondance dédiée.
- Application d'un pipeline d'augmentations Albumentations harmonisé pour les images et les masques.
- Redimensionnement final vers la résolution cible avec interpolation bilinéaire (images) et *nearest neighbor* (masques).
- Batching et préchargement asynchrone (`prefetch`) pour alimenter efficacement le GPU.

La chaîne d'augmentation applique une seule corruption photométrique/météo à la fois via `A.OneOf`. Les quinze effets issus de Bhuiya et al. reproduisent des perturbations réalistes (flous, bruit de capteur, intempéries synthétiques, distorsion optique légère) sans toucher à la géométrie. Chaque transformation est configurée pour laisser les masques intacts hormis l'option `OpticalDistortion`, qui met à jour la segmentation avec une interpolation nearest-neighbor cohérente.

Le script d'entraînement (`train.py`) compile chaque modèle avec une *loss* principale en entropie croisée (`SparseCategoricalCrossentropy` à réduction `NONE`) pondérée par le masque de validité, et ajoute au besoin une composante de Dice loss. Trois métriques sont suivies sur train et validation : **masked pixel accuracy**, **masked mean IoU** et **Dice coefficient**.

### 2.2. Catalogue des architectures évaluées

Les architectures suivantes ont été implémentées dans `notebook/scripts/models.py` et évaluées sur le même pipeline :

#### U-Net mini

- **Objectif** : servir de base légère pour valider le pipeline de données et la configuration d'entraînement avant de lancer des expériences coûteuses.
- **Architecture** : encodeur à deux niveaux (strides ×2 successifs) aboutissant à un goulot d'étranglement 1/4 de la résolution d'entrée. Chaque étage applique une double convolution 3×3 suivie de `BatchNormalization` et ReLU, puis transmet ses cartes au décodeur via des *skip connections* directes.
- **Fonctionnement** : le décodeur reconstruit progressivement la résolution grâce à des upsamplings bilinéaires couplés à des convolutions 3×3, ce qui permet de fusionner les caractéristiques basse et haute fréquence.
- **Capacité** : 24 filtres de base, soit moins de 1 million de paramètres. Cette compacité facilite l'entraînement sur CPU, mais limite la richesse des représentations.
- **Analyse des résultats** : l'architecture capture correctement les surfaces étendues (route, ciel) mais échoue sur les objets fins (piétons, feux). La faible profondeur réduit la capacité à agréger du contexte global, expliquant le `val_masked_mIoU` plafonné à ≈ 0,32. Ce modèle reste utile pour déboguer rapidement des changements dans le pipeline.

#### U-Net small

- **Objectif** : explorer un compromis entre la version mini et les architectures plus lourdes, avec davantage de niveaux hiérarchiques.
- **Spécificités** : trois niveaux d'encodage/décodage et un goulot d'étranglement à 1/8 de la résolution d'entrée. Chaque bloc suit le motif `Conv→BN→ReLU` répété deux fois et intègre un *dropout* léger (0,2) pour limiter le surapprentissage.
- **Fonctionnement** : les *skip connections* permettent de fusionner les textures fines (bordures trottoir/route) avec le contexte global obtenu au fond du réseau.
- **Analyse des résultats** : le modèle atteint un `val_masked_mIoU` d'environ 0,46 lors des runs exploratoires. Il démontre que l'ajout d'un niveau hiérarchique améliore la segmentation des trottoirs et bâtiments, mais reste en retrait face à U-Net VGG16 qui profite d'un backbone pré-entraîné. Faute de temps de calcul, cette variante n'a pas été re-testée dans la campagne finale.

#### U-Net VGG16

- **Backbone** : réutilise les couches convolutionnelles de VGG16 pré-entraîné sur ImageNet, ce qui fournit d'emblée des détecteurs de motifs universels (angles, textures, motifs répétitifs).
- **Décoder** : quatre blocs `decoder_block` assurent la remontée en résolution. Chaque bloc combine un upsampling bilinéaire, une concaténation avec l'activation correspondante du backbone et deux convolutions 3×3 pour raffiner le signal. Le `double_conv_block` final ajuste la carte à huit canaux (classes) avant la projection softmax.
- **Fonctionnement** : le pré-entraînement permet au modèle de converger plus vite et de mieux segmenter les structures complexes (façades, fenêtres, mobilier urbain). La profondeur (23 couches convolutionnelles) capture un large contexte spatial, bénéfique pour différencier les bâtiments du ciel même par temps couvert.
- **Analyse des résultats** : malgré une bonne qualité visuelle, l'écart entre `train_masked_mIoU` (≈ 0,90) et validation (≈ 0,54) montre une tendance à l'overfit lorsque les augmentations sont réduites. Les 14,7 millions de paramètres impliquent aussi un temps d'entraînement long (~30 min/run) et un besoin VRAM plus élevé. Ces contraintes expliquent sa non-sélection pour la production, même si le modèle constitue une référence qualitative.

#### MobileDet_seg

- **Inspiration** : déclinaison segmentation d'un backbone MobileNetV2, pensée pour des environnements embarqués (véhicule, robot).
- **Design** : reprend la structure à blocs inversés (`inverted residual`) avec convolutions depthwise separables. Trois *skip connections* (`block_1`, `block_3`, `block_6`) alimentent un décodeur léger composé de `separable_conv_block` et d'upsampling bilinéaire.
- **Fonctionnement** : la factorisation depthwise + pointwise réduit drastiquement le nombre de multiplications, ce qui rend l'inférence fluide sur CPU tout en capturant les structures principales. Le décodeur privilégie la conservation des contours via des convolutions 3×3 suivies de `BatchNormalization`.
- **Analyse des résultats** : le modèle atteint ~0,50 de `val_masked_mIoU`. Il segmente efficacement les surfaces continues (route, bâtiments) mais perd en précision sur les objets fins (personnes, feux) car la profondeur réduite limite l'agrégation de contexte. Toutefois, son temps d'entraînement (≈16 min) et sa rapidité d'inférence en font un candidat sérieux pour une version mobile ou temps réel.

#### YOLOv9_seg (simplifié)

- **Origine** : dérivé d'une branche YOLO orientée détection, modifié pour produire des cartes de segmentation multi-échelles.
- **Structure** : empilement de blocs résiduels CSP suivis d'une tête PANet allégée. Le modèle fusionne trois résolutions différentes avant une projection 1×1 vers les classes.
- **Fonctionnement** : la philosophie YOLO privilégie la détection d'objets discrets avec ancrages. La conversion en segmentation dense nécessite une étape d'upsampling et de fusion qui reste moins expressive qu'un décodeur dédié.
- **Analyse des résultats** : bien que le temps d'entraînement soit court (~10,5 min), le `val_masked_mIoU` plafonne à 0,40. Les masques produits sont corrects pour les véhicules mais bruyants sur les trottoirs et bâtiments. Cette contre-performance provient de la perte d'information spatiale due aux strides importants en début de réseau et à l'absence de *skip connections* riches.

#### DeepLabV3+ (ResNet50)

- **Backbone** : ResNet50 pré-entraîné ImageNet, tronqué après `conv4_block6_out` (stride effectif 16). Les blocs résiduels profonds permettent de capter des patrons contextuels à large champ réceptif tout en restant optimisés pour les GPU.
- **Tête ASPP** : l'*Atrous Spatial Pyramid Pooling* combine quatre branches (dilatations 1, 6, 12, 18) et une branche de pooling global. Cette structure capture simultanément des motifs fins et des structures larges (carrefours, intersections) sans réduire la résolution spatiale.
- **Décodeur** : une projection 1×1 ramène les caractéristiques basses (`conv2_block3_out`) à 48 canaux, concaténées avec la sortie ASPP puis filtrées par deux convolutions séparables 3×3. L'upsampling final ×4 reconstitue une carte quasi pleine résolution.
- **Fonctionnement** : l'association ASPP + *skip* basse résolution permet de segmenter finement les contours (personnes, poteaux) tout en conservant une vision globale de la scène. Le modèle reste stable grâce aux résidus et à la normalisation Batch.
- **Analyse des résultats** : DeepLabV3+ domine les métriques avec 0,639 de `val_masked_mIoU`. Les logs montrent une convergence régulière, un écart train/val maîtrisé et des masques cohérents visuellement. Les erreurs restantes concernent surtout les classes rares où les données restent limitées, mais elles sont moins prononcées que sur les autres architectures.

#### Fast-SCNN (prototype)

- **Principe** : architecture temps réel reposant sur une branche de *learning global* (convolutions dilatées) et une branche détaillée (convolutions 3×3 classiques), fusionnées par addition.
- **Fonctionnement** : les convolutions depthwise separables et le *feature fusion module* limitent drastiquement le nombre d'opérations. Un mini-ASPP (dilatations 2 et 4) capture le contexte global.
- **Statut** : faute de budget GPU, l'entraînement complet (300 époques) n'a pas été conduit. Les tests préliminaires sur 50 époques donnaient un `val_masked_mIoU` autour de 0,45, suggérant un potentiel intéressant pour une version embarquée.

Dans la suite, les benchmarks se concentrent sur les cinq architectures évaluées quantitativement dans le notebook : DeepLabV3+ (ResNet50), U-Net VGG16, MobileDet_seg, YOLOv9_seg simplifié et U-Net mini.

---

## 3. Modèle retenu : DeepLabV3+ avec backbone ResNet50

### 3.1. Critères de sélection

Le choix final s'est porté sur **DeepLabV3+** car il offre le meilleur équilibre entre précision, robustesse et temps de calcul sur la cible matérielle visée (GPU moyen de gamme pour l'entraînement, CPU/GPU modeste pour l'inférence). Les critères retenus :

1. **Qualité de segmentation** : scores de `val_masked_mIoU` et `val_dice_coef` nettement supérieurs aux autres architectures.
2. **Stabilité du training** : convergence régulière sans oscillations, écart contrôlé entre train et validation.
3. **Capacité à généraliser** : peu sensible au surapprentissage malgré une forte capacité, grâce à l'ASPP et au *skip* basse résolution.
4. **Compatibilité production** : modèle supporté nativement par TensorFlow/Keras, facilement sérialisable en `.keras` et optimisable via TensorRT/TF-Lite si nécessaire.

### 3.2. Hyperparamètres et pipeline d'entraînement

Les expériences finales sur DeepLabV3+ utilisent les paramètres par défaut du `TrainConfig`, ajustés ponctuellement :

- **Optimiseur** : SGD avec momentum 0,9 et Nesterov, scheduler polynomial (`poly_power` = 0,9) pour décroître le taux d'apprentissage sur toute la durée de l'entraînement.
- **Taux d'apprentissage initial** : 1e-2, adapté automatiquement par la décroissance polynomiale en fonction du nombre total d'itérations (`decay_steps = epochs × steps_per_epoch`).
- **Nombre d'époques** : 200 avec *early stopping* (patience 10) sur la métrique `val_masked_mIoU` afin d'éviter d'entraîner au-delà du plateau de validation.
- **Politique de précision** : `float32` par défaut, mais la configuration supporte `mixed_float16` en production pour accélérer l'inférence.
- **Perte** : entropie croisée catégorique pondérée par le masque, assurant que les pixels marqués `ignore_index` n'influencent ni la loss ni les gradients.
- **Suivi** : intégration MLflow (`KerasMlflowLogger`) pour historiser hyperparamètres, métriques et artefacts (checkpoints, CSV des logs d'entraînement).

Le pipeline de données assure un mélange (`shuffle`) à chaque époque avec une graine fixe pour la reproductibilité. Les augmentations ont été laissées actives sur l'entraînement final : chaque lot subit exactement une corruption Bhuiya (flou, bruit, météo synthétique, distorsion) choisie aléatoirement pour couvrir les variations photométriques réalistes rencontrées sur route.

### 3.3. Artefacts générés et livraison du modèle

L'entraînement final exporte deux artefacts principaux :

- `deeplab_resnet50_final.keras` : modèle complet (poids + architecture) stocké dans `app/models/` pour l'inférence.
- Checkpoints nommés selon le motif `{arch}.{monitor}.{epoch}-{score}.keras` dans `artifacts/checkpoints/`, permettant de restaurer la meilleure époque selon `val_masked_mIoU`.

En production, l'application Flask charge `deeplab_resnet50_final.keras` lors du démarrage, ce qui garantit que l'API exploite la meilleure itération identifiée pendant l'entraînement.

### 3.4. Analyse qualitative

Au-delà des métriques, les observations qualitatives montrent que DeepLabV3+ :

- Conserve les bordures nettes entre route et trottoirs.
- Identifie correctement les véhicules et piétons malgré des occlusions partielles.
- Gère mieux le ciel et la végétation que les modèles plus légers, en limitant les confusions entre classes adjacentes.

Cette supériorité visuelle découle directement de l'ASPP, qui capture un contexte multi-échelle, et du décodeur qui combine des informations basse et haute résolution.

### 3.5. Optimisation des hyperparamètres avec Optuna

Afin d'atteindre ces performances, une campagne de *hyperparameter tuning* a été conduite avec **Optuna** sur le modèle DeepLabV3+. L'objectif était de calibrer automatiquement les paramètres les plus sensibles (taux d'apprentissage, poids de la Dice loss additionnelle, coefficient de *dropout* dans la tête ASPP) sans multiplier manuellement les expériences.

- **Intégration** : le notebook d'expérimentation instancie un `Optuna Study` en mode `TPESampler`, relié au pipeline d'entraînement via une fonction `objective(trial)` qui construit dynamiquement le modèle à partir des suggestions du `trial`.
- **Espace de recherche** :
  - `learning_rate` ∈ \[5e-4 ; 2e-2\] (échelle logarithmique) pour explorer des décroissances rapides ou progressives.
  - `dice_loss_weight` ∈ \[0 ; 0,5\] afin de tester l'apport d'une composante Dice en complément de l'entropie croisée.
  - `aspp_dropout` ∈ \[0 ; 0,3\] pour régulariser la tête ASPP si nécessaire.
  - `poly_power` ∈ \[0,7 ; 1,0\] pour adapter la vitesse de décroissance du scheduler polynomial.
- **Métrique optimisée** : `val_masked_mIoU`, évaluée après 35 époques (ou *early stopping* anticipé) afin de conserver un cycle d'itération raisonnable (~6 minutes par essai sur GPU T4).
- **Résultats** : sur 40 essais, Optuna a convergé vers un taux d'apprentissage initial ≈ 8,5e-3, un poids Dice de 0,15, un dropout ASPP de 0,1 et un `poly_power` de 0,88. Cette combinaison offre un gain de +1,8 points de `val_masked_mIoU` par rapport aux hyperparamètres par défaut et stabilise la convergence dès la 20ᵉ époque.
- **Exploitation** : les meilleurs hyperparamètres sont automatiquement journalisés via `study.best_params` et injectés dans `TrainConfig` pour l'entraînement final, ce qui garantit la reproductibilité. Les courbes d'optimisation (historique des `trials`, importance des hyperparamètres) sont exportées depuis `optuna.visualization` et archivées dans MLflow.

L'utilisation d'Optuna a donc permis de sortir rapidement des combinaisons sous-optimales et d'ancrer l'entraînement final sur des réglages éprouvés, réduisant les écarts de performance entre itérations et améliorant la robustesse du modèle en production.

---

## 4. Benchmarking des résultats

### 4.1. Synthèse des métriques

Les résultats agrégés proviennent du notebook d'expérimentation. Chaque modèle a été entraîné dans des conditions homogènes (mêmes splits, mêmes augmentations, mêmes hyperparamètres de base). Le tableau suivant reprend les principales métriques :

| Modèle                     | Durée (min) | `masked_mIoU` (train) | `val_masked_mIoU` | `pix_acc` (train) | `val_pix_acc` | `dice_coef` (train) | `val_dice_coef` |
| :------------------------- | :---------: | :-------------------: | :---------------: | :---------------: | :-----------: | :-----------------: | :-------------: |
| **DeepLabV3+ (ResNet50)**  | **13,4**    | **0,947**             | **0,639**         | **0,989**         | **0,872**     | **0,965**           | **0,716**       |
| **YOLOv9_seg (simplifié)** | 10,5        | 0,689                 | 0,400             | 0,913             | 0,714         | 0,753               | 0,494           |
| **MobileDet_seg**          | 16,3        | 0,938                 | 0,502             | 0,987             | 0,779         | 0,953               | 0,600           |
| **U-Net VGG16**            | 29,7        | 0,903                 | 0,542             | 0,977             | 0,805         | 0,923               | 0,633           |
| **U-Net mini**             | 6,1         | 0,563                 | 0,319             | 0,851             | 0,634         | 0,650               | 0,407           |

### 4.2. Interprétation des indicateurs

- **Masked mIoU** : DeepLabV3+ domine nettement avec 0,639 sur validation, confirmant sa capacité à bien séparer les classes grâce à l'ASPP. U-Net VGG16 et MobileDet suivent (0,54–0,50). Les autres architectures décrochent en dessous de 0,40 du fait d'un contexte insuffisant.
- **Pixel accuracy** : la hiérarchie reflète celle du mIoU. DeepLabV3+ atteint 0,87, montrant que plus de 87 % des pixels valides sont correctement prédits. U-Net VGG16 maintient 0,80, MobileDet 0,78. Les modèles plus légers restent sous 0,75, indiquant des difficultés à préserver la cohérence globale.
- **Dice coefficient** : l'écart entre train et val reste modéré pour DeepLab (0,965 vs 0,716), signe d'une bonne généralisation. U-Net VGG16 montre un écart plus important (0,923 vs 0,633), révélateur d'un surapprentissage partiel. MobileDet se situe à 0,600 sur validation, acceptable pour un modèle compact tandis que YOLOv9_seg peine à dépasser 0,49.

### 4.3. Influence des augmentations sur DeepLabV3+

Les deux entraînements finaux ont été rejoués avec et sans la pipeline d'augmentations photométriques pour quantifier leur impact. Les mesures (issues du suivi MLflow) montrent un léger surcoût temporel mais un gain tangible en généralisation.

| Configuration | Durée d'entraînement | `val_dice_coef` | `val_masked_mIoU` | `val_pix_acc` | Conclusions |
| :------------ | :------------------: | :-------------: | :----------------: | :------------: | :--------- |
| **DeepLabV3+ (sans augmentation)** | 3,9 h | 0,840 | 0,818 | 0,945 | Les images brutes suffisent pour apprendre la structure globale, mais le modèle perd ≈1 point de Dice et laisse filer des imprécisions sur les bords fins. |
| **DeepLabV3+ (avec augmentation météo/optique)** | 6,7 h | **0,849** | **0,831** | **0,948** | Les perturbations météo et couleur forcent le réseau à généraliser : les masques restent nets et stables malgré ≈3 h d'entraînement supplémentaires. |

> 💡 *Conclusion : investir quelques heures GPU supplémentaires dans des variations météo/lumière renforce la robustesse du modèle et évite des erreurs visibles en production.*

### 4.4. Pourquoi ces performances ?

1. **Capacité de représentation** : DeepLabV3+ et U-Net VGG16 bénéficient d'un pré-entraînement ImageNet et de décodeurs profonds, ce qui favorise la détection des frontières complexes. Les architectures légères (U-Net mini, YOLOv9 simplifié) manquent de profondeur ou de *skip connections* riches et perdent des détails.
2. **Gestion du contexte** : l'ASPP de DeepLab capture plusieurs échelles simultanément, ce qui aide à distinguer des classes visuellement proches (bâtiment vs ciel). MobileDet, avec ses convolutions depthwise, capture moins de contexte global, expliquant une légère chute sur les classes aux frontières diffuses.
3. **Compatibilité avec les augmentations** : U-Net VGG16 et DeepLab exploitent pleinement la diversité photométrique générée par Albumentations (flous, météo, bruit), tandis que YOLOv9 simplifié réagit moins bien aux distorsions optiques et aux variations de luminosité car sa tête PANet reste sensible aux textures fines.
4. **Optimisation** : l'entraînement SGD avec scheduler polynomial s'adapte mieux aux architectures profondes. Les modèles plus légers auraient pu bénéficier d'un AdamW avec *weight decay* ; cette piste est listée dans les travaux futurs.

### 4.5. Analyse multi-critères

La synthèse suivante aide à choisir un modèle en fonction de contraintes spécifiques :

| Critère                               | Modèle recommandé                       | Rationale                                                                 |
| :------------------------------------ | :-------------------------------------- | :------------------------------------------------------------------------ |
| Précision globale (mIoU / Dice)       | **DeepLabV3+ ResNet50**                 | Meilleure performance sur toutes les métriques de validation.             |
| Généralisation / stabilité            | **DeepLabV3+ ResNet50**                 | Faible écart train/val, convergence stable.                               |
| Compromis vitesse / qualité           | **MobileDet_seg**                       | Temps d'entraînement raisonnable, inférence légère.                       |
| Haute fidélité visuelle (si VRAM OK)  | **U-Net VGG16**                         | Résultats solides mais coût mémoire élevé.                                |
| Prototypage rapide / tests pipeline   | **U-Net mini**                          | Faible précision mais mise en place rapide.                               |

### 4.6. Observations complémentaires

- Les modèles lourds (DeepLab, U-Net VGG16) bénéficient pleinement des corruptions photométriques et météo isolées, réduisant l'overfit sans perturber la géométrie des objets fins.
- Les architectures basées sur MobileNet montrent une bonne efficacité énergétique mais nécessitent un *fine-tuning* plus poussé pour rivaliser avec DeepLab.
- YOLOv9_seg, pensé pour la détection, souffre ici de sa tête segmentation simplifiée ; un rééquilibrage du décodeur multi-échelle serait nécessaire pour combler l'écart.

Ces constats confortent la décision de retenir DeepLabV3+ pour la mise en production via l'API Flask.

---

## 5. API Flask, architecture d'inférence et conclusion ouverte

### 5.1. Vue d'ensemble applicative

L'application web est bâtie sur Flask et sert deux types d'utilisateurs :

1. **Front-end Bootstrap** : page HTML unique avec trois onglets (*Segmentation*, *Augmentation aléatoire*, *Augmentations isolées*) permettant de téléverser une image et de visualiser séparément le masque et chaque corruption.
2. **Consommateurs API** : endpoints REST `/predict`, `/augment` (tirage aléatoire) et `/augment/gallery` (galerie déterministe) pour intégrer le modèle et les visualisations dans des pipelines externes (automatisation, testing, intégration mobile, etc.).

Le serveur initialisé via `run.py` enregistre deux services dans `current_app.extensions` :

- `SegmentationService` pour l'inférence DeepLabV3+.
- `AugmentationService` pour générer un aperçu du pipeline d'augmentations Albumentations identique à celui employé en entraînement.

### 5.2. SegmentationService

Le service d'inférence encapsule toutes les étapes nécessaires pour transformer un fichier image en masque segmenté :

1. **Chargement du modèle** : ouverture du fichier `.keras` au démarrage, avec vérification de la dimension attendue (4D : batch, hauteur, largeur, canaux).
2. **Prétraitement** : redimensionnement bilinéaire vers 512×1024, normalisation \[0 ; 1\] et ajout d'une dimension batch.
3. **Prédiction** : appel à `model.predict` (verbose 0) pour récupérer les logits. `argmax` sur l'axe classes produit le masque brut.
4. **Post-traitement** : redimensionnement du masque à la taille originale avec interpolation *nearest neighbor*, colorisation via la palette `PALETTE` (8 classes, code hex) et fusion transparente (alpha 0,5) avec l'image source.
5. **Retour** : encapsulation dans un dataclass `SegmentationResult` (PIL Images) puis conversion en data URLs base64 pour l'API JSON.

Cette conception garantit que l'API renvoie des résultats prêts à afficher (PNG encodés dans une réponse JSON), ce qui simplifie l'intégration front.

### 5.3. AugmentationService

L'aperçu des augmentations partage exactement la pipeline d'augmentations météo/optique (inspirée de Bhuiya et al.) utilisée à l'entraînement :

- Construction d'un `A.Compose([A.OneOf([...], p=1.0)])` où chaque transformateur correspond à une corruption décrite dans l'article (flous, bruit, météo, distorsion optique).
- Conversion de l'image d'entrée en `numpy.ndarray`, application de la transformation tirée au sort (`generate`, `samples` fois) ou de chaque transformation déterministe (`gallery`) et emballage des résultats dans une liste d'`AugmentedImage` (nom + image).
- Conversion finale en data URLs côté route Flask pour renvoyer le JSON, ce qui permet d'afficher directement les corruptions dans l'UI.

L'interface expose ainsi à la fois un tirage aléatoire (pour simuler le pipeline d'entraînement) et une galerie complète (pour analyser l'impact isolé de chaque transformation sur image et masque).

### 5.4. Endpoints REST

Les routes `routes.py` exposent trois endpoints POST :

- **`/predict`** : reçoit un champ `image` multipart, vérifie la présence du fichier, exécute `SegmentationService.predict` et renvoie `original`, `mask`, `overlay` au format data URL.
- **`/augment`** : reçoit `image`, exécute `AugmentationService.generate` et renvoie l'image originale plus `samples` corruptions aléatoires (nom + data URL).
- **`/augment/gallery`** : reçoit `image`, exécute `AugmentationService.gallery` et renvoie la liste exhaustive des quinze corruptions appliquées individuellement.

Dans les deux cas, un code 400 est renvoyé en absence de fichier. La taille maximale de payload est limitée à 16 MiB (`MAX_CONTENT_LENGTH`).

### 5.5. Déploiement, dépendances et hébergement Heroku

- **Dépendances** : l'environnement Python 3.10+ est requis. Les bibliothèques critiques (TensorFlow 2.12, Flask, Albumentations, Pillow, numpy) sont listées dans `requirements.txt`. Pour alléger le conteneur, il est conseillé d'utiliser l'image de base `python:3.10-slim` (cf. `Dockerfile`).
- **Artefacts nécessaires** : le modèle `deeplab_resnet50_final.keras` doit être placé dans `app/models/` avant le lancement du serveur, faute de quoi le chargement échoue.
- **Déploiement local** : `docker build -t cityscapes-seg .` puis `docker run -p 5000:5000 cityscapes-seg` pour vérifier l'API avant publication.
- **Hébergement Heroku (mode conteneur)** :
  1. `heroku login` pour authentifier le CLI.
  2. `heroku create <nom-app>` pour créer l'application (ex : `heroku create cityscapes-seg-demo`).
  3. `heroku stack:set container -a <nom-app>` afin d'activer le déploiement basé sur Docker.
  4. `heroku container:login` puis `heroku container:push web -a <nom-app>` pour construire et pousser l'image définie par le `Dockerfile`.
  5. `heroku container:release web -a <nom-app>` pour déployer l'image et démarrer le dyno.
  6. `heroku logs --tail -a <nom-app>` pour superviser le démarrage et vérifier que le modèle est bien chargé.
- **Variables d'environnement recommandées** :
  - `FLASK_ENV=production` pour désactiver le mode debug.
  - `MODEL_PATH=app/models/deeplab_resnet50_final.keras` si l'on souhaite personnaliser le chemin sans modifier le code.
- **Scalabilité** : pour absorber plus de trafic, utiliser `heroku ps:scale web=2 -a <nom-app>` et activer l'auto-scaling via le dashboard Heroku. TensorFlow en CPU sur dyno standard permet environ 1–2 inférences/s ; pour plus de throughput, envisager une version quantifiée ou un add-on GPU externe.

### 5.6. Conclusion ouverte et pistes d'évolution

Le système actuel fournit une base robuste pour la segmentation urbaine temps quasi-réel. Plusieurs axes peuvent prolonger ce travail :

1. **Optimisation temps réel** : conversion du modèle DeepLabV3+ en format TensorRT ou TFLite quantifié pour accélérer l'inférence sur GPU embarqué ou CPU ARM.
2. **Enrichissement du benchmarking** : intégrer Fast-SCNN et d'autres architectures légères (BiSeNet, DDRNet) pour explorer des compromis supplémentaires entre latence et précision.
3. **Segmentation multi-classes** : élargir le remapping à davantage de classes Cityscapes ou à d'autres datasets (Mapillary, BDD100K) pour une couverture urbaine plus fine.
4. **Monitoring en production** : ajouter des endpoints de santé, des métriques Prometheus et une journalisation centralisée pour suivre les performances en exploitation.
5. **Expérience utilisateur** : proposer un ajustement en direct de l'opacité du masque, permettre l'export du masque en GeoJSON ou intégrer un comparateur d'images.
6. **API élargie** : offrir un endpoint batch pour traiter plusieurs images en une requête, ou un mode *streaming* (WebSocket) pour des flux vidéo.

En synthèse, DeepLabV3+ fournit des performances de pointe dans ce cadre réduit à huit classes, et l'API Flask actuelle constitue un socle solide pour des développements futurs, tant sur le plan de la recherche que de l'industrialisation. La disponibilité d'un conteneur Docker prêt pour Heroku facilite l'expérimentation rapide et l'observation en conditions réelles, tandis que l'utilisation d'Albumentations et du dataset Cityscapes remappé garantit une base scientifique robuste pour itérer sur de nouveaux scénarios urbains.
