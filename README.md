# Food-11 — Classification d'images de nourriture (MATLAB)

Projet de deep learning en vision par ordinateur : classer des photos de plats en **11 catégories**
(bread, dairy, dessert, egg, fried, meat, noodles-pasta, rice, seafood, soup, vegetable-fruit),
comparer plusieurs réseaux et soumettre les prédictions du meilleur sur le jeu de test.

Le classement du challenge utilise le **Frugality Score** :

```
Frugality = F1 − 0.3 / (1 + 1/R)        R = nombre de paramètres en millions
```

Un bon modèle est donc à la fois **précis** et **léger**.

## Arborescence

```
Food11/
├── setup_project.m          ← à lancer en premier (path + vérifications)
├── data/
│   ├── train/               ← 13 296 images, 11 sous-dossiers (non versionné)
│   ├── test/                ← 3 097 images sans label (non versionné)
│   └── splits/              ← split stratifié 80/20 (seed 42)
│       ├── train_files.csv        10 637 images
│       └── validation_files.csv    2 659 images
├── src/                     ← fonctions réutilisées par les scripts
│   ├── load_split.m               CSV → imageDatastore
│   ├── leNet5.m                   architecture LeNet-5
│   └── makeCourseProgressOutputFcn.m   courbes d'entraînement
├── preprocessing/           ← analyse et préparation du dataset
│   ├── analyze_dataset.m
│   ├── split_dataset.m
│   ├── preprocess_images.m
│   └── augment_images.m
├── training/                ← entraînement des réseaux
│   ├── train_lenet5.m
│   └── train_resnet18.m
├── models/                  ← réseaux entraînés (.mat)
│   ├── lenet5_model.mat
│   ├── mobilenetv2_model.mat
│   └── resnet18_model.mat
├── evaluation/              ← comparaison et soumission
│   ├── evaluate_models.m
│   ├── compare_models.m
│   └── generate_json.m
├── results/
│   ├── preprocessing/       ← figures + normalization_stats.mat
│   └── evaluation/          ← métriques, matrices de confusion, JSON (généré)
└── docs/
    ├── preprocessing.md     ← analyse du dataset et choix de prétraitement
    ├── evaluation.md        ← métriques, figures et procédure de soumission
    └── lenet5_integration.docx
```

## Prérequis

- MATLAB R2025a, **Deep Learning Toolbox**
- Le dataset Food-11 copié localement dans `data/train/<classe>/*.jpg` et `data/test/*.jpg`

## Utilisation

```matlab
setup_project                 % une fois par session

% 1. Prétraitement (déjà fait : résultats dans results/preprocessing et data/splits)
analyze_dataset
split_dataset                 % régénère les CSV (même seed → même split)
preprocess_images             % calcule mu / sigma
augment_images                % visualise les augmentations

% 2. Entraînement (les modèles sont déjà fournis dans models/)
train_lenet5
train_resnet18

% 3. Évaluation sur la validation puis soumission
evaluate_models
compare_models
generate_json                 % → results/evaluation/submission_food11.json
```

Tous les scripts retrouvent seuls la racine du projet : aucun chemin absolu n'est codé en dur.

## Modèles

| Modèle | Approche | Entrée | Paramètres (≈) |
|---|---|---|---|
| LeNet-5 | CNN entraîné de zéro | 128×128×3 | 3,75 M |
| MobileNetV2 | Transfer learning ImageNet | 224×224×3 | 2,2 M |
| ResNet-18 | Transfer learning ImageNet (dernière couche seule) | 224×224×3 | 11,2 M |

- **Même split pour tous** : les trois modèles sont entraînés et évalués sur les CSV de `data/splits/`.
- **Normalisation** : LeNet-5 utilise les statistiques du train (`results/preprocessing/normalization_stats.mat`) ;
  les réseaux pré-entraînés gardent leur normalisation ImageNet intégrée.
- **Validation et test** : redimensionnement uniquement, jamais d'augmentation.

## Détails

- [docs/preprocessing.md](docs/preprocessing.md) — analyse du dataset (déséquilibre, dimensions), split, normalisation, augmentation
- [docs/evaluation.md](docs/evaluation.md) — métriques (macro-F1, balanced accuracy…), figures du rapport, format du JSON
