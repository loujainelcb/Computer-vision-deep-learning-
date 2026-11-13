%% Data augmentation (uniquement sur le train)
% Les parametres sont definis UNE seule fois dans src/trainingConfig.m
% et utilises a la fois ici (visualisation) et dans les trois
% entrainements (src/makeDatastores.m).
clear; clc; close all;

sz = 128;

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));
resultsDir = fullfile(root, 'results', 'preprocessing');
cfg = trainingConfig();
a = cfg.aug;

%% Définition des augmentations
augmenter = makeAugmenter(cfg);

%% Datastores réellement utilisés en entraînement
[dsTrain, dsVal, imdsTrain] = makeDatastores(root, [sz sz], cfg);

%% Visualisation sur une image
rng(42);
I = imresize(imread(imdsTrain.Files{1}), [sz sz]);

tests = {
    'Rotation',    imageDataAugmenter('RandRotation', a.rotation)
    'Flip',        imageDataAugmenter('RandXReflection', a.reflectionX)
    'Zoom',        imageDataAugmenter('RandScale', a.scale)
    'Translation', imageDataAugmenter('RandXTranslation', a.translation, 'RandYTranslation', a.translation)
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
            J = randomBrightness(I, a.brightness);
            nom = 'Luminosité';
        end
        subplot(nTests, 5, 5*(t-1) + k + 1); imshow(J); title(nom);
    end
end
saveas(gcf, fullfile(resultsDir, 'augmentation_examples.png'));

%% Vérification : augmentation sur le train uniquement
a1 = read(dsTrain); reset(dsTrain); b1 = read(dsTrain);
fprintf('Train : deux lectures identiques ? %d (doit être 0)\n', isequal(a1.input, b1.input));
a1 = read(dsVal); reset(dsVal); b1 = read(dsVal);
fprintf('Validation : deux lectures identiques ? %d (doit être 1)\n', isequal(a1.input, b1.input));
