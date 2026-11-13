%% Data augmentation (uniquement sur le train)
clear; clc; close all;

sz = 128;

% Chargement du split
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));
trainDir = fullfile(root, 'data', 'train');
resultsDir = fullfile(root, 'results', 'preprocessing');
imdsTrain = load_split(fullfile(root, 'data', 'splits', 'train_files.csv'), trainDir);
imdsVal = load_split(fullfile(root, 'data', 'splits', 'validation_files.csv'), trainDir);

%% Définition des augmentations
augmenter = imageDataAugmenter( ...
    'RandRotation', [-15 15], ...           % rotation légère
    'RandXReflection', true, ...            % flip horizontal
    'RandScale', [0.9 1.2], ...             % zoom
    'RandXTranslation', [-10 10], ...       % translation (pixels)
    'RandYTranslation', [-10 10]);

%% Visualisation sur une image
rng(42);
I = imresize(imread(imdsTrain.Files{1}), [sz sz]);

tests = {
    'Rotation',    imageDataAugmenter('RandRotation', [-15 15])
    'Flip',        imageDataAugmenter('RandXReflection', true)
    'Zoom',        imageDataAugmenter('RandScale', [0.9 1.2])
    'Translation', imageDataAugmenter('RandXTranslation', [-10 10], 'RandYTranslation', [-10 10])
    'Toutes',      augmenter
};

figure('Position', [50 50 1000 1200]);
nTests = size(tests, 1) + 1;   % + la luminosité
for t = 1:nTests
    subplot(nTests, 5, 5*(t-1) + 1); imshow(I); title('Originale');
    for k = 1:4
        if t <= size(tests, 1)
            J = augment(tests{t, 2}, I);
            nom = tests{t, 1};
        else
            J = changerLuminosite(I);
            nom = 'Luminosité';
        end
        subplot(nTests, 5, 5*(t-1) + k + 1); imshow(J); title(nom);
    end
end
saveas(gcf, fullfile(resultsDir, 'augmentation_examples.png'));

%% Datastores pour l'entraînement
% train : avec augmentation (+ luminosité)
dsTrain = augmentedImageDatastore([sz sz], imdsTrain, 'DataAugmentation', augmenter);
dsTrain = transform(dsTrain, @luminositeBatch);

% validation : SANS augmentation, seulement redimensionnée
dsVal = augmentedImageDatastore([sz sz], imdsVal);

% Vérification
a = read(dsTrain); reset(dsTrain); b = read(dsTrain);
fprintf('Train : deux lectures identiques ? %d (doit être 0)\n', isequal(a.input, b.input));
a = read(dsVal); reset(dsVal); b = read(dsVal);
fprintf('Validation : deux lectures identiques ? %d (doit être 1)\n', isequal(a.input, b.input));

%% Fonctions
function J = changerLuminosite(I)
% facteur aléatoire entre 0.8 et 1.2
J = uint8(double(I) * (0.8 + 0.4*rand));
end

function data = luminositeBatch(data)
for i = 1:height(data)
    data.input{i} = changerLuminosite(data.input{i});
end
end
