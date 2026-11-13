# P1 — Preprocessing du dataset Food-11 (MATLAB)

Analyse, split, prétraitement et data augmentation du train set Food-11
(13 296 images, 11 classes).

## Prérequis

- MATLAB R2025a (testé), **Deep Learning Toolbox**
- Images Food-11 dans `data/train/<classe>/*.jpg`


## Exécution

Depuis MATLAB, lancer `setup_project` à la racine, puis les scripts de `preprocessing/` dans cet ordre (ils retrouvent seuls la racine du projet) :

| Ordre | Script | Rôle | Sorties |
|---|---|---|---|
| 1 | `analyze_dataset.m` | Comptage par classe, déséquilibre, exemples, anomalies | `results/preprocessing/class_distribution.png`, `results/preprocessing/sample_images.png` |
| 2 | `split_dataset.m` | Split stratifié 80/20, seed = 42 | `data/splits/train_files.csv`, `data/splits/validation_files.csv` |
| 3 | `preprocess_images.m` | Resize 128/256, format MATLAB, normalisation | `results/preprocessing/preprocessing_comparison.png`, `results/preprocessing/normalization_stats.mat` |
| 4 | `augment_images.m` | Augmentations (train uniquement) | `results/preprocessing/augmentation_examples.png` |

Les scripts 3 et 4 lisent les CSV produits par le script 2.

## 1. Analyse du dataset

- **13 296 images**, **11 classes** : bread, dairy, dessert, egg, fried, meat,
  noodles-pasta, rice, seafood, soup, vegetable-fruit.

| Classe | Images | % |
|---|---:|---:|
| dessert | 2000 | 15.0 |
| soup | 2000 | 15.0 |
| meat | 1774 | 13.3 |
| bread | 1356 | 10.2 |
| egg | 1313 | 9.9 |
| seafood | 1202 | 9.0 |
| fried | 1174 | 8.8 |
| vegetable-fruit | 941 | 7.1 |
| noodles-pasta | 587 | 4.4 |
| dairy | 573 | 4.3 |
| rice | 376 | 2.8 |

- **Déséquilibre** : ratio max/min = **5,3** (dessert/soup 2000 vs rice 376),
  coefficient de variation 0,47. Un modèle qui prédit toujours la classe
  majoritaire obtient 15 % d'accuracy. À cause de ce déséquilibre, on suit
  aussi l'**accuracy équilibrée** et le **macro-F1**. On peut aussi pondérer
  la loss par classe.
- **Anomalies** :
  - aucune image corrompue ;
  - 100 % JPEG, toutes en RGB (3 canaux) ;
  - **695 dimensions différentes** : de 207 à 6144 px en hauteur et de 220 à
    9216 px en largeur (médiane 512×512) ;
  - ratio largeur/hauteur entre 0,56 et 2,61 ;
  - aucune image n'a un côté < 128 px ; 8 images ont un côté < 256 px.

  → le redimensionnement est obligatoire.

## 2. Split Train / Validation

- `splitEachLabel(imds, 0.8, 'randomized')` découpe **chaque classe** séparément,
  ce qui donne un split stratifié.
- `rng(42, 'twister')` rend le split reproductible.
- **10 637 train / 2 659 validation**. L'écart de proportion entre train et validation est d'au plus 0,02 point.
- Les CSV contiennent le chemin relatif de chaque image (`data/train/<classe>/<image>.jpg`) et sa classe (`File`, `Label`).
  Toutes les expériences utilisent donc exactement le même split. Si le
  dossier est déplacé, il suffit de relancer `split_dataset.m` (même seed,
  donc même split).

## 3. Preprocessing

| | 128×128 | 256×256 |
|---|---|---|
| Format MATLAB | `128×128×3×N`, `uint8` | `256×256×3×N`, `uint8` |
| Lecture + resize | ~3,3 ms/image | ~4,1 ms/image |
| Train complet en mémoire | 0,52 Go | 2,09 Go |

- **Redimensionnement** : `augmentedImageDatastore([sz sz], ...)` fait un resize
  direct, sans conserver le ratio. On garde ainsi l'image entière. La plupart
  des images sont carrées (médiane 1,0), donc la déformation reste faible.
- **Format attendu par MATLAB** : `H×W×C×N` (format `SSCB`), `uint8`, 3 canaux,
  labels `categorical`. `ColorPreprocessing = 'gray2rgb'` sert de sécurité.
- **Normalisation** : z-score par canal, avec des statistiques calculées **sur
  le train uniquement** :
  - moyenne RGB = [142.00, 115.21, 87.86] ;
  - écart-type RGB = [68.09, 68.80, 70.45].

  Sur la validation, on obtient une moyenne de 0,05 et un écart-type de 0,97.
  La normalisation est intégrée au réseau, donc elle s'applique à l'identique
  au train, à la validation et au test :

  ```matlab
  load results/preprocessing/normalization_stats.mat   % mu, sigma
  imageInputLayer([sz sz 3], 'Normalization', 'zscore', 'Mean', mu, 'StandardDeviation', sigma)
  ```

## 4. Data augmentation

| Transformation | Paramètre MATLAB | Plage |
|---|---|---|
| Rotation légère | `RandRotation` | ±15° |
| Flip horizontal | `RandXReflection` | oui |
| Zoom | `RandScale` | ×0,9 – ×1,2 |
| Translation | `RandX/YTranslation` | ±10 pixels |
| Luminosité | `transform` + fonction `randomBrightness` | facteur ×0,8 – ×1,2 |

- Ces plages sont définies **une seule fois** dans `src/trainingConfig.m` (`cfg.aug`).
  `augment_images.m` les visualise, et les trois entraînements les utilisent via
  `src/makeDatastores.m` : modifier une plage la change partout.
- `imageDataAugmenter` n'a pas d'option de luminosité. Elle est donc ajoutée
  par `transform(...)` sur le datastore d'entraînement.
- **Les augmentations ne sont appliquées qu'au train.** Le datastore de
  validation n'a pas d'augmenter. Le script le vérifie :
  - deux lectures d'un même batch de train donnent des images différentes ;
  - deux lectures d'un même batch de validation donnent des images identiques.
- La rotation, le zoom arrière et la translation laissent des bords noirs
  (remplissage à 0).

```matlab
% Utilisé tel quel par les trois scripts d'entraînement (via runTraining)
[dsTrain, dsVal] = makeDatastores(root, [sz sz]);   % train augmenté, val sans augmentation
```

## 5. Expérimentations prévues

Sur ce même split, les expériences à mener :

- **résolution** : 128 vs 256 ;
- **data augmentation** : avec vs sans ;
- **taille du jeu d'entraînement** : 25 / 50 / 100 % du train, sous-échantillonné de façon stratifiée.

On compare l'accuracy de validation, l'accuracy équilibrée, le macro-F1 et
l'écart train/validation (indicateur de sur-apprentissage).
