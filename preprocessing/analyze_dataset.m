%% Analyse du dataset Food-11 (train)
clear; clc; close all;

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));
trainDir   = fullfile(root, 'data', 'train');
resultsDir = fullfile(root, 'results', 'preprocessing');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

%% Chargement des images
imds = imageDatastore(trainDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
classes = categories(imds.Labels);

fprintf('Nombre d''images : %d\n', numel(imds.Files));
fprintf('Nombre de classes : %d\n', numel(classes));

%% Nombre d'images par classe
counts = countEachLabel(imds);
disp(counts);

% Déséquilibre entre les classes
fprintf('Classe max : %d images, classe min : %d images\n', max(counts.Count), min(counts.Count));
fprintf('Ratio max/min : %.2f\n', max(counts.Count) / min(counts.Count));

figure;
bar(counts.Label, counts.Count);
ylabel('Nombre d''images');
title('Nombre d''images par classe');
grid on;
saveas(gcf, fullfile(resultsDir, 'class_distribution.png'));

%% Exemples de chaque catégorie
rng(42);
nEx = 4;
figure('Position', [20 20 1650 680]);
tiledlayout(nEx, numel(classes), 'TileSpacing', 'tight');
for c = 1:numel(classes)
    idx = find(imds.Labels == classes{c});
    idx = idx(randperm(numel(idx), nEx));
    for k = 1:nEx
        nexttile((k-1)*numel(classes) + c);
        imshow(imresize(imread(imds.Files{idx(k)}), [200 200]));
        if k == 1, title(classes{c}); end
    end
end
saveas(gcf, fullfile(resultsDir, 'sample_images.png'));

%% Vérification des anomalies
n = numel(imds.Files);
H = zeros(n,1); W = zeros(n,1); C = zeros(n,1);
nCorrupt = 0;
warning('off', 'all');   % warnings EXIF inutiles sur certains JPEG
for i = 1:n
    try
        I = imread(imds.Files{i});
        [H(i), W(i), C(i)] = size(I);
    catch
        nCorrupt = nCorrupt + 1;
        fprintf('Image corrompue : %s\n', imds.Files{i});
    end
end
warning('on', 'all');

fprintf('Images corrompues : %d\n', nCorrupt);
fprintf('Images non RGB : %d\n', sum(C ~= 3));
fprintf('Dimensions différentes : %d\n', size(unique([H W], 'rows'), 1));
fprintf('Hauteur : min %d, max %d\n', min(H), max(H));
fprintf('Largeur : min %d, max %d\n', min(W), max(W));
