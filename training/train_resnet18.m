
%% FOOD-11 - Transfer Learning ResNet-18

%% 1. Initialisation
clc;
clear;
rng(42);

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(projectRoot, 'src'));

%% 2. Charger ResNet-18 pre-entraine

netResNet = imagePretrainedNetwork( ...
    "resnet18", NumClasses=11);

disp("ResNet-18 charge avec succes !");

%% 3. Chargement du dataset Food-11

trainDir = fullfile(projectRoot, 'data', 'train');

trainCSV = fullfile(projectRoot, 'data', 'splits', 'train_files.csv');
valCSV = fullfile(projectRoot, 'data', 'splits', 'validation_files.csv');

imdsTrain = load_split(trainCSV, trainDir);
imdsVal = load_split(valCSV, trainDir);

fprintf('Images entrainement : %d\n', numel(imdsTrain.Files));
fprintf('Images validation : %d\n', numel(imdsVal.Files));

%% 4. Preparation des images pour ResNet-18

inputSize = [224 224];

augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal = augmentedImageDatastore(inputSize, imdsVal);

disp("Images preparees pour ResNet-18 !");

%% 5. Configuration de l'entrainement

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 200, ...
    'ValidationPatience', 5, ...
    'Metrics', 'accuracy', ...
    'OutputNetwork', 'best-validation', ...
    'ExecutionEnvironment', 'gpu', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

%% 6. Verification des couches de classification

disp(netResNet.Learnables(end-5:end, :));

%% 7. Gel des couches pre-entrainees

learnables = netResNet.Learnables;

% Adapter le nom si la derniere couche est differente
lastLayerName = learnables.Layer(end);

for i = 1:height(learnables)

    layerName = learnables.Layer(i);
    parameterName = learnables.Parameter(i);

    if layerName == lastLayerName
        factor = 1;
    else
        factor = 0;
    end

    netResNet = setLearnRateFactor( ...
        netResNet, layerName, parameterName, factor);
end

disp("Couche de classification entrainable :");
disp(lastLayerName);

%% 8. Entrainement ResNet-18

disp("Debut de l'entrainement ResNet-18...");

tic;

[netResNetTrained, infoResNet] = trainnet( ...
    augTrain, netResNet, "crossentropy", options);

trainingTime = toc;

fprintf('Temps d''entrainement : %.2f minutes\n', ...
    trainingTime / 60);

disp("Entrainement termine !");

%% 9. Sauvegarde du modele

modelsDir = fullfile(projectRoot, 'models');

if ~exist(modelsDir, 'dir')
    mkdir(modelsDir);
end

classNames = categories(imdsTrain.Labels);   % ordre des sorties du reseau
inputSize = [inputSize 3];

save(fullfile(modelsDir, 'resnet18_model.mat'), ...
    'netResNetTrained', 'infoResNet', 'classNames', 'inputSize', ...
    'trainingTime', '-v7.3');

disp("Modele ResNet-18 sauvegarde !");
