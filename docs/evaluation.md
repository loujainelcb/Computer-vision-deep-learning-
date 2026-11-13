# Food-11 — P4 : évaluation, comparaison et JSON final

**Version préparatoire du 11 novembre 2026 — MATLAB R2025a / Deep Learning Toolbox.**

Ce dossier contient **trois scripts MATLAB autonomes**, adaptés aux scripts P1/P2 fournis et aux deux CSV réels. Ils ne modifient aucun fichier de P1/P2 et **n'entraînent pas** les réseaux. Ils ne calculent **jamais** une métrique sur le test officiel non étiqueté.

> **État de vérification :** les deux CSV fournis ont été contrôlés indépendamment : 10 637 images de train, 2 659 de validation, 11 catégories, aucun doublon ni recouvrement. En revanche, les images du dataset et les modèles entraînés `.mat` n'étaient pas disponibles lors de la création : **les scripts n'ont pas encore été exécutés dans MATLAB sur de vrais modèles**.

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

## 3. Ce que P2 et P3 doivent mettre dans `models/`

| Modèle | Nom de fichier recommandé | Entrée probable |
|---|---|---|
| P2 LeNet-5 | `lenet5_model.mat` | 128 × 128 × 3 |
| P3 MobileNetV2 | `mobilenetv2_model.mat` | souvent 224 × 224 × 3 |
| P3 ResNet-18 | `resnet18_model.mat` | 224 × 224 × 3 |

**Pour P2** : son fichier `mai_train_lenet5.m` prévoit déjà :

```matlab
save(fullfile(modelsDir,'lenet5_model.mat'),'trainedNet','trainInfo','inputSize','classNames')
```

**Pour P3** : enregistrer le **réseau réellement entraîné à 11 classes**, pas le réseau ImageNet à 1000 classes. Si le modèle est un `dlnetwork` issu de `trainnet`, enregistrer **en plus** l'ordre des catégories de sortie :

```matlab
classNames = categories(imdsTrain.Labels);   % si c'est l'ordre utilisé
inputSize  = [224 224 3];
save('models/mobilenetv2_model.mat','net','classNames','inputSize')
```

Même chose pour ResNet. L'ordre `classNames` doit correspondre exactement aux colonnes des scores du réseau. Si P3 a appliqué un prétraitement supplémentaire **en dehors du modèle** (normalisation personnalisée, recadrage...), il faudra le reproduire dans `predictLabels` : les scripts actuels redimensionnent les images et laissent le réseau effectuer sa **normalisation interne**, sans data augmentation lors de la validation/test.

**En cas d'erreur** : remplacer `executionEnvironment='auto'` par `'cpu'` si le GPU n'est pas utilisable, et réduire `miniBatchSize=32` à `16` ou `8` si nécessaire.

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
- **`learning_curve_*.png`** : si P2/P3 sauvegardent un `trainInfo` avec TrainingAccuracy / ValidationAccuracy ; utile pour discuter du surapprentissage.
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

Le challenge permet des évaluations privées si on **ne coche pas « Update leaderboard ? »**. Mettre à jour le leaderboard seulement pour l'entrée finale du groupe. L'entrée peut être écrasée si on réutilise le même nom de groupe.

## 6. Ce qui reste à faire dans le projet collectif

- **P1** : fournir aux machines locales `data/train` et `data/test` ; le split CSV est déjà disponible.
- **P2** : résoudre si besoin les dépendances de son script d'entraînement (ex. `makeCourseProgressOutputFcn`), puis sauvegarder LeNet-5 entraîné en `.mat`.
- **P3** : entraîner MobileNetV2 et ResNet, sauvegarder les modèles et, pour `dlnetwork`, les `classNames`.
- **P4** : lancer les trois scripts, vérifier les matrices de confusion, argumenter le meilleur compromis F1/paramètres ; préparer JSON et soumission.

**Note** : les scripts n'ont pas pu être lancés dans MATLAB ici (MATLAB et les modèles ne sont pas disponibles dans cet environnement). Il faudra éventuellement adapter leurs entrées à la manière exacte dont P3 a entraîné/sauvegardé ses réseaux. Ils sont fournis comme **base de travail commentée et reproductible**, pas comme résultats déjà vérifiés.
