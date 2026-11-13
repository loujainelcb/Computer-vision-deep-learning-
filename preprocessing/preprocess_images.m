%% Preprocessing : redimensionnement et normalisation
clear; clc; close all;


% Chargement du split
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));
trainDir = fullfile(root, 'data', 'train');
resultsDir = fullfile(root, 'results', 'preprocessing');
imdsTrain = load_split(fullfile(root, 'data', 'splits', 'train_files.csv'), trainDir);

%% Redimensionnement 128x128 et 256x256 + format attendu par MATLAB
for sz = [128 256]
    ds = augmentedImageDatastore([sz sz], imdsTrain, 'ColorPreprocessing', 'gray2rgb');
    batch = read(ds);
    X = cat(4, batch.input{:});
    fprintf('Taille %d : batch de taille %s, type %s\n', sz, mat2str(size(X)), class(X));
    % format attendu : H x W x 3 x N, uint8
end

%% Normalisation : moyenne et écart-type par canal sur le train
ds = augmentedImageDatastore([128 128], imdsTrain);
ds.MiniBatchSize = 256;
somme = zeros(1,1,3); somme2 = zeros(1,1,3); nPix = 0;
while hasdata(ds)
    batch = read(ds);
    X = double(cat(4, batch.input{:}));
    somme  = somme  + sum(X, [1 2 4]);
    somme2 = somme2 + sum(X.^2, [1 2 4]);
    nPix   = nPix + size(X,1) * size(X,2) * size(X,4);
end
mu    = somme / nPix;
sigma = sqrt(somme2 / nPix - mu.^2);
fprintf('Moyenne RGB : %.1f %.1f %.1f\n', mu);
fprintf('Ecart-type RGB : %.1f %.1f %.1f\n', sigma);
save(fullfile(resultsDir, 'normalization_stats.mat'), 'mu', 'sigma');
% Utilisation dans le réseau :
% imageInputLayer([sz sz 3], 'Normalization', 'zscore', 'Mean', mu, 'StandardDeviation', sigma)

%% Figure de comparaison
rng(42);
figure('Position', [50 50 1200 900]);
for i = 1:4
    I = imread(imdsTrain.Files{randi(numel(imdsTrain.Files))});
    I128 = imresize(I, [128 128]);
    In = (double(I128) - mu) ./ sigma;   % image normalisée

    subplot(4, 4, 4*(i-1) + 1); imshow(I);                 title('Originale');
    subplot(4, 4, 4*(i-1) + 2); imshow(I128);              title('128x128');
    subplot(4, 4, 4*(i-1) + 3); imshow(imresize(I, [256 256])); title('256x256');
    subplot(4, 4, 4*(i-1) + 4); imshow(rescale(In));       title('Normalisée');
end
saveas(gcf, fullfile(resultsDir, 'preprocessing_comparison.png'));
