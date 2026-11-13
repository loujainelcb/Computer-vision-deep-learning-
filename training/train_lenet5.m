%=============================================
% SCRIPT PRINCIPAL - ENTRAINEMENT DE LENET-5 
%===============================================

clc; clear; close all;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));



% ---  CHARGEMENT DES DONNÉES  ---
fprintf('1. Chargement du split train/validation...\n');
csvTrain = fullfile(root, 'data', 'splits', 'train_files.csv');
csvVal   = fullfile(root, 'data', 'splits', 'validation_files.csv');
localTrainDir = fullfile(root, 'data', 'train');

imdsTrain = load_split(csvTrain, localTrainDir);
imdsVal  = load_split(csvVal, localTrainDir);



%------------Prétraitement et augmentation de données ---
inputSize = [128 128 3]; %Dimension choisie

%Data Augmentation (Rotation léger, flp horizontal)
augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandRotation', [-15 15] );
    
% Chargement des statistiques de normalisation  (si disponibles)
normFile = fullfile(root, 'results', 'preprocessing', 'normalization_stats.mat');

if exist(normFile, 'file')
    load(normFile, 'mu', 'sigma');
    fprintf('   -> Statistiques de normalisation chargées : utilisation de la normalisation Z-score (mu et sigma).\n');
    inputLayer = imageInputLayer(inputSize, Name ='input', ...
        Normalization='zscore', ...
        Mean = mu, ...
        StandardDeviation=sigma);
else
   fprintf('-> Fichier normalization_stats.mat non trouvé : bascule sur la normalisation [0, 1].\n');
   
    inputLayer = imageInputLayer(inputSize, Name= 'input', ...
        Normalization = 'rescale-zero-one');
end
%Augmentation du datastores en redimensonnant les images

augImdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, DataAugmentation=augmenter);
augImdsVal= augmentedImageDatastore(inputSize(1:2), imdsVal);

%----Récuperation de lenet5
numClasses=numel(categories(imdsTrain.Labels));
layers=leNet5(inputSize,numClasses, inputLayer);

%---Config paramètres---
% il faut changer 'ExecutionEnvironment' sur 'cpu' pour les tests locaux sans GPU
% Avec Gpu, mettre 'ExecutionEnvironment sur 'gpu' 

maxEpochs= 20;
miniBatchSize = 64;
smoothingWindow=10;
labelsTrain= imdsTrain.Labels;
progressFcn = makeCourseProgressOutputFcn( ...
    numel(labelsTrain), miniBatchSize, maxEpochs, smoothingWindow);
%Adam permet un réajustement auto de chaque paraametre
%Mettre verbose à false pour enlever les log pendant l'execution
options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=maxEpochs, ...
    Shuffle="every-epoch", ...
    MiniBatchSize=miniBatchSize, ...
    ValidationData=augImdsVal, ...
    ValidationFrequency=50, ...
    Verbose=true, ...
    Plots="training-progress", ...
    OutputFcn=progressFcn, ...
    Metrics="accuracy",...
    ExecutionEnvironment='cpu') ; %cpu ou gpu

% ---LANCEMENT DE L'ENTRAÎNEMENT ---
fprintf('Lancement de l''entraînement de LeNet-5...\n');
[trainedNet, trainInfo] = trainnet(augImdsTrain,layers,"crossentropy", options);

% ---  SAUVEGARDE DU MODÈLE  ---
modelsDir = fullfile(root, 'models');
if ~exist(modelsDir, 'dir'), mkdir(modelsDir); end
classNames = categories(imdsTrain.Labels);   % ordre des sorties du reseau
save(fullfile(modelsDir, 'lenet5_model.mat'), 'trainedNet', 'trainInfo', 'inputSize', 'classNames');
fprintf('Entraînement terminé ! Modèle sauvegardé dans models/lenet5_model.mat\n');