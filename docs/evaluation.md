# Food-11 — Évaluation, comparaison et JSON final

MATLAB R2025a / Deep Learning Toolbox.

Le dossier `evaluation/` contient **trois scripts autonomes**. Ils **n'entraînent pas** les réseaux et ne calculent **jamais** de métrique sur le test officiel non étiqueté.

> **Contrôle du split :** 10 637 images de train, 2 659 de validation, 11 catégories, aucun doublon ni recouvrement (vérifié aussi à chaque exécution de `evaluate_models.m`).

## 1. Emplacement

Les trois scripts sont dans `evaluation/`. L'arborescence complète du projet est décrite dans le `README.md` à la racine.

**Important :** `data/train` et `data/test` ne sont pas versionnés : chaque machine doit les remplir localement.

## 2. Exécuter les scripts — dans cet ordre

**Dans MATLAB, à la racine du projet**, lancer `setup_project`, puis :

```matlab
% 1. Dès qu'un premier modèle entraîné est disponible :
evaluate_models

% 2. Pour comparer les modèles disponibles :
compare_models

% 3. Quand les trois modèles sont évalués :
generate_json
```

- `evaluate_models.m` **ignore poliment** les modèles `.mat` qui manquent et compare ceux qui sont disponibles ; il crée une matrice de confusion et des statistiques détaillées pour chaque modèle.
- `compare_models.m` crée les graphiques pour le rapport, même si un ou deux modèles manquent, mais marque la comparaison comme **provisoire**.
- `generate_json.m` **refuse par défaut** de finaliser tant que les **trois modèles** ne sont pas dans le tableau d'évaluation. Pour un essai provisoire **uniquement**, modifier en haut `requireAllThreeModels=false`.

## 3. Format des modèles dans `models/`

| Modèle | Fichier | Entrée | Produit par |
|---|---|---|---|
| LeNet-5 | `lenet5_model.mat` | 128 × 128 × 3 | `training/train_lenet5.m` |
| MobileNetV2 | `mobilenetv2_model.mat` | 224 × 224 × 3 | `training/train_mobilenetv2.m` |
| ResNet-18 | `resnet18_model.mat` | 224 × 224 × 3 | `training/train_resnet18.m` |

Les trois scripts d'entraînement passent par `src/runTraining.m`, qui sauvegarde toujours les mêmes variables :

```matlab
save(..., 'net','trainInfo','classNames','inputSize','trainingTime','cfg')
```

- `classNames` : ordre exact des 11 sorties du réseau. S'il manque (anciens modèles), l'évaluation suppose l'ordre alphabétique, qui est celui de `trainnet`, et affiche un avertissement.
- `cfg` : le protocole d'entraînement utilisé (`src/trainingConfig.m`), pour pouvoir le citer dans le rapport.

Toute la normalisation est **dans le réseau** : l'évaluation se contente de redimensionner les images, sans augmentation, exactement comme le datastore de validation pendant l'entraînement.

**En cas d'erreur** : remplacer `executionEnvironment='auto'` par `'cpu'` si le GPU n'est pas utilisable, et réduire `miniBatchSize=32` à `16` ou `8` si nécessaire (sans effet sur les prédictions).

## 4. Les comparaisons — ce que demande vraiment le cours et le challenge

Les supports fournis indiquent :

- **Challenge PDF, pages 1-2** : entraîner, valider et analyser **au moins deux réseaux** ; soumettre les prédictions du meilleur sur le test. Le critère de classement est le **Frugality Score** avec `w = 0.3` et `R` = paramètres **en millions**.
- **CVML_complementary_slides, pages 4 et 6** : bien séparer train / validation / test ; comprendre précision, rappel et F1-score.
- **CVML_lectures, pages 39, 53 à 58** : analyser les performances après entraînement, utiliser `ValidationData`, surveiller l'accuracy en validation au cours des epochs.
- **P1 README** : dataset déséquilibré, donc compléter l'accuracy par des métriques par classe et le **macro-F1**.

### Métriques calculées par `evaluate_models.m`

| Métrique | Interprétation rapide |
|---|---|
| **Accuracy** | Pourcentage global d'images correctement classées. |
| **Précision par classe** | Parmi les prédictions « soupe », combien sont réellement des soupes ? |
| **Rappel par classe** | Parmi les vraies soupes, combien ont été reconnues ? |
| **F1 par classe** | Compromis entre précision et rappel. |
| **Macro-F1** | Moyenne des F1 des **11 classes** avec le même poids ; intéressant avec votre déséquilibre. |
| **Weighted-F1** | Moyenne pondérée par le nombre d'images de chaque classe ; complément. |
| **Balanced accuracy** | Moyenne des rappels des 11 classes. |
| **Paramètres (M)** | Nombre de paramètres du réseau, en millions. |
| **Frugality Score** | `macroF1 - 0.3/(1 + 1/ParametersM)`. Plus élevé = meilleur compromis. |

**Nuance :** le challenge parle de « F1 » et de « performances moyennes entre classes » pour le serveur, sans donner explicitement son code d'agrégation dans le document fourni. **Le macro-F1 est donc choisi comme estimation locale** ; le Frugality Score officiel sera celui renvoyé par le serveur. Le temps d'inférence est également mesuré pour information, mais **n'entre pas** dans la formule du professeur.

### Figures générées (à insérer dans le rapport)

- **`comparison_dashboard.png`** : une seule grande figure présentant **(a)** Accuracy / macro-F1 / balanced accuracy, **(b)** Frugality Score, **(c)** millions de paramètres, **(d)** nuage F1 vs paramètres. Les échelles sont séparées pour ne pas mélanger paramètres et scores.
- **`classwise_f1_heatmap.png`** : tableau coloré 11 classes × modèles, pour comprendre qui reconnaît le mieux le riz, les desserts, etc.
- **`confusion_leNet_5.png`**, **`confusion_mobilenetv2.png`**, **`confusion_resnet.png`** : matrices de confusion normalisées **par vraie classe** (les colonnes représentent les classes prédites).
- **`learning_curve_*.png`** : accuracy train / validation au fil des itérations (lue dans `trainInfo`) ; utile pour discuter du surapprentissage.
- **`top_errors_*.csv`** : cinq confusions les plus fréquentes pour argumenter le rapport.

Les figures *communes* répondent à « quel réseau gagne et pourquoi ? » ; les matrices *séparées* expliquent « où se trompe chaque réseau ? ». Le professeur n'impose pas un graphique unique : c'est un choix de présentation.

## 5. Comment reconnaître le bon fichier JSON ?

`generate_json.m` écrit `results/evaluation/submission_food11.json`, avec **3 097 entrées** du type :

```json
{
  "0": "bread",
  "1": "dessert",
  "2": "fried"
}
```

**L'exemple ci-dessus est fictif**, les vraies prédictions dépendront du modèle entraîné. Le script utilise les vrais noms de fichiers sans extension comme clés (ne suppose pas que les images sont numérotées sans trous).

Le script crée aussi `test_predictions.csv` et `test_class_distribution.png` **pour contrôle interne** : ne pas les envoyer au serveur. Le site demande le **JSON**, le nombre de paramètres **en millions** et une description courte du modèle.
Pour tester un modèle particulier sans écraser la soumission automatique, définir
`FOOD11_MODEL="ResNet-18 FT"` avant de lancer `generate_json.m`. Le fichier créé sera
alors `results/evaluation/submission_food11_resnet_18_ft.json`. Les valeurs possibles
sont `LeNet-5`, `MobileNetV2`, `ResNet-18 FT` (fine-tuning complet) et `ResNet-18 gele`.

Le challenge permet des évaluations privées si on **ne coche pas « Update leaderboard ? »**. Mettre à jour le leaderboard seulement pour l'entrée finale du groupe. L'entrée peut être écrasée si on réutilise le même nom de groupe.

## 6. Points à discuter dans le rapport

- **Comparaison équitable** : les trois modèles partagent le même split, la même augmentation, les mêmes poids de classes, le même budget (epochs, optimiseur) et la même règle de sélection (`best-validation`). Seules diffèrent la taille d’entrée, la normalisation et le learning rate (imposés par le point de départ : zéro ou ImageNet), ainsi que la taille de batch, choisie par modèle après essais.
- **Biais de validation** : la validation sert à choisir le meilleur epoch et le meilleur modèle ; les scores locaux sont donc légèrement optimistes. Le score du serveur, sur le test, est la référence.
- **Frugalité** : avec `freezeBackbone = true`, peu de poids sont entraînés, mais **tous** comptent dans le nombre de paramètres R.
