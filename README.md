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
├── run_evaluation.m         ← évaluation + figures + tableau + JSON en une commande
├── data/
│   ├── train/               ← 13 296 images, 11 sous-dossiers (non versionné)
│   ├── test/                ← 3 097 images sans label (non versionné)
│   └── splits/              ← split stratifié 80/20 (seed 42)
│       ├── train_files.csv        10 637 images
│       └── validation_files.csv    2 659 images
├── src/                     ← fonctions partagées
│   ├── trainingConfig.m           protocole d'entraînement COMMUN aux 3 modèles
│   ├── runTraining.m              entraînement + sauvegarde (utilisé par les 3)
│   ├── makeDatastores.m           split + redimensionnement + augmentation
│   ├── makeAugmenter.m            augmentations géométriques de P1
│   ├── randomBrightness.m         augmentation de luminosité de P1
│   ├── classWeights.m             poids de classes (déséquilibre)
│   ├── freezeBackbone.m           gel des couches pré-entraînées
│   ├── makeCourseProgressOutputFcn.m   courbes d'entraînement (mêmes pour les 3)
│   ├── load_split.m               CSV → imageDatastore
│   └── leNet5.m                   architecture LeNet-5
├── preprocessing/           ← analyse et préparation du dataset
│   ├── analyze_dataset.m
│   ├── split_dataset.m
│   ├── preprocess_images.m
│   └── augment_images.m
├── training/                ← un script court par modèle
│   ├── train_lenet5.m
│   ├── train_mobilenetv2.m        entraînement direct (GPU conseillé)
│   ├── tune_transfer.m            recherche d'hyperparamètres + modèle final (CPU)
│   ├── train_resnet18.m           ResNet-18 gelé
│   ├── train_resnet18_partial_ft.m  fine-tuning res5a/res5b + tête
│   └── train_resnet18_full_ft.m     fine-tuning complet
├── models/                  ← réseaux entraînés (.mat)
│   ├── lenet5_model.mat
│   ├── mobilenetv2_model.mat          modèle soumis
│   ├── resnet18_model.mat             ResNet-18 gelé
│   ├── resnet18_partial_ft[_final].mat
│   └── resnet18_full_ft[_final].mat
├── evaluation/              ← comparaison et soumission
│   ├── evaluate_models.m
│   ├── compare_models.m
│   ├── make_presentation.m        tableau + figures pour les slides
│   └── generate_json.m
├── results/
│   ├── preprocessing/       ← figures + normalization_stats.mat
│   ├── training/            ← courbes accuracy / loss de chaque modèle
│   ├── tuning/              ← recherche d'hyperparamètres (csv, figure, log)
│   ├── evaluation/          ← métriques, matrices de confusion, JSON (généré)
│   └── presentation/        ← tableau récapitulatif + toutes les figures des slides
└── docs/
    ├── preprocessing.md     ← analyse du dataset et choix de prétraitement
    └── evaluation.md        ← métriques, figures et procédure de soumission
```

## Comment les parties sont liées

```
preprocessing/  ──►  data/splits/*.csv ─────────────┐
                ──►  normalization_stats.mat ─► LeNet│
src/trainingConfig.m (augmentation, epochs, …) ─────┤
                                                    ▼
training/train_*.m ──► src/runTraining.m ──► models/*.mat
                                                    │
evaluation/evaluate_models ◄── même split de validation
          → compare_models → generate_json → submission_food11.json
```

Les paramètres d'augmentation sont définis **une seule fois** dans `src/trainingConfig.m`. Ils servent à la fois à la visualisation de P1 (`augment_images.m`) et aux trois entraînements.

## Prérequis

- MATLAB R2025a avec la Deep Learning Toolbox (sans elle les réseaux .mat ne se chargent pas)
- Support packages MobileNet-v2 et ResNet-18, seulement pour réentraîner
- Le dataset Food-11 copié dans `data/train/<classe>/*.jpg` et `data/test/*.jpg` (pas sur git)

## Refaire la soumission sans réentraîner

Les modèles entraînés sont dans `models/`. Dans MATLAB, à la racine du projet :

```matlab
setup_project
run_evaluation
```

Le JSON est créé dans `results/evaluation/submission_food11.json`. Le modèle utilisé est celui
qui a le meilleur Frugality Score sur la validation (MobileNetV2).

## Utilisation

```matlab
setup_project                 % une fois par session

% 1. Prétraitement (déjà fait : résultats dans results/preprocessing et data/splits)
analyze_dataset
split_dataset                 % régénère les CSV (même seed → même split)
preprocess_images             % calcule mu / sigma
augment_images                % visualise les augmentations

% 2. Entraînement (protocole commun : src/trainingConfig.m)
train_lenet5
tune_transfer("mobilenetv2")  % recherche de réglages + modèle final (CPU, environ 5 h)
train_resnet18                % ResNet-18 gelé
train_resnet18_partial_ft     % fine-tuning partiel (GPU)
train_resnet18_full_ft        % fine-tuning complet (GPU)

% 3. Évaluation, figures, tableau et JSON en une commande :
run_evaluation
% = evaluate_models, compare_models, make_presentation, generate_json
make_report_figures           % figures du rapport
```

Pour relancer un fine-tuning, il faut d'abord supprimer ou renommer `models/resnet18_*_ft*.mat`
(le script ne les écrase pas).

| Sortie | Fichier |
|---|---|
| Tableau des 4 modèles | `results/presentation/tableau_recapitulatif.png` (+ `.csv`, `.md`) |
| Matrices de confusion | `results/presentation/confusion_3_modeles.png` |
| Accuracy / loss par epoch | `results/presentation/courbes_validation_3_modeles.png` |
| F1 par classe | `results/presentation/f1_par_classe.png` |
| Dataset, augmentation, progression, dashboard | `results/presentation/0*_*.png` |
| **Soumission** | `results/evaluation/submission_food11.json` |

Tous les scripts retrouvent seuls la racine du projet : aucun chemin absolu n'est codé en dur.

## Modèles

| Modèle | Architecture | Point de départ | Entrée | Paramètres (≈) |
|---|---|---|---|---|
| LeNet-5 | 4 conv + 1 denses | Poids aléatoires | 128×128×3 | 2,52 M |
| MobileNetV2 | Convolutions séparables, blocs résiduels inversés | ImageNet | 224×224×3 | 2,2 M |
| ResNet-18 | 18 couches à connexions résiduelles | ImageNet | 224×224×3 | 11,2 M |

### Protocole de comparaison

Base commune (`src/trainingConfig.m`) :

| Élément | Valeur |
|---|---|
| Split | `data/splits/` — stratifié 80/20, seed 42 |
| Augmentation (train) | rotation ±15°, flip horizontal, zoom ×0,9–1,2, translation ±10 px, luminosité ×0,8–1,2 |
| Déséquilibre | entropie croisée pondérée par l'inverse de la fréquence des classes |
| Budget | 15 epochs, Adam, une validation par epoch |
| Courbes | même figure accuracy / loss (lissée sur 10 itérations) → `results/training/` |
| Modèle retenu | meilleur epoch en validation |
| Seed | 42 |
| Validation / test | redimensionnement seul, aucune augmentation |

Réglages de chaque modèle :

| | LeNet-5 | MobileNetV2 (`tune_transfer`) | ResNet-18 gele (`train_resnet18`) | ResNet-18 FT (`train_resnet18_full_ft`) |
|---|---|---|---|---|
| Stratégie | from scratch | extraction de caractéristiques | backbone gelé | fine-tuning complet |
| Taille d'entrée | 128×128 | 224×224 | 224×224 | 224×224 |
| Normalisation | z-score avec les stats du train | ImageNet intégrée | ImageNet intégrée | ImageNet intégrée |
| Couches entraînées | toutes | dernière couche (14 091 paramètres) | dernière couche | toutes, à partir de ResNet-18 gele |
| Learning rate | 1e-3 | 3e-4 (recherche) | 1e-4 | 1e-5 |
| Batch | 64 | 32 (recherche) | 8 | 8 |
| Régularisation | dropout 0,2 + L2 par défaut (1e-4) | L2 1e-5 + dropout 0,2 (recherche) | L2 par défaut (1e-4) | L2 par défaut (1e-4) |
| Pondération des classes | oui | non (recherche) | oui | oui |
| Augmentation | à la volée | 2 copies pré-calculées par image | à la volée | à la volée |
| Epochs | 15 | 12 max. | 15 | 10 |

Pour MobileNetV2, la recherche d'hyperparamètres (32 configurations, critère macro-F1) est dans
`results/tuning/`.

Les modèles pré-entraînés nécessitent les support packages MATLAB *Deep Learning Toolbox Model for ResNet-18 Network* et *… for MobileNet-v2 Network*.

## Détails

- [docs/preprocessing.md](docs/preprocessing.md) — analyse du dataset (déséquilibre, dimensions), split, normalisation, augmentation
- [docs/evaluation.md](docs/evaluation.md) — métriques (macro-F1, balanced accuracy…), figures du rapport, format du JSON
