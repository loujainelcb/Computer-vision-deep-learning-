%% ENTRAINEMENT LENET-5 (CNN entraine de zero)
% Protocole commun : src/trainingConfig.m
% Specifique a ce modele : entree 128x128, normalisation z-score avec les
% statistiques du train calculees par P1, learning rate "scratch".

clc; clear; close all;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));

cfg = trainingConfig();
rng(cfg.seed);

inputSize = [128 128 3];

% Normalisation : mu / sigma du train (preprocessing/preprocess_images.m)
normFile = fullfile(root, 'results', 'preprocessing', 'normalization_stats.mat');
if ~isfile(normFile)
    error('%s absent : lancer preprocessing/preprocess_images.m d''abord.', normFile);
end
load(normFile, 'mu', 'sigma');
inputLayer = imageInputLayer(inputSize, Name='input', ...
    Normalization='zscore', Mean=mu, StandardDeviation=sigma);

net = dlnetwork(leNet5(inputSize, 11, inputLayer));

runTraining(net, inputSize, cfg.learnRate.scratch, 'lenet5_model.mat', root, cfg);
