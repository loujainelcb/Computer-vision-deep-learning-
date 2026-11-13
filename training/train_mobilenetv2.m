%% ENTRAINEMENT MOBILENETV2 (transfer learning ImageNet)
% Protocole commun : src/trainingConfig.m
% Specifique a ce modele : entree 224x224 et normalisation ImageNet
% integrees au reseau pre-entraine, learning rate "transfer".
% Necessite le support package "Deep Learning Toolbox Model for MobileNet-v2 Network".

clc; clear; close all;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));

cfg = trainingConfig();
rng(cfg.seed);

% Reseau ImageNet dont la derniere couche est remplacee par 11 classes
net = imagePretrainedNetwork("mobilenetv2", NumClasses=11);
inputSize = net.Layers(1).InputSize;   % [224 224 3]

if cfg.freezeBackbone
    net = freezeBackbone(net);
end

runTraining(net, inputSize, cfg.learnRate.transfer, 'mobilenetv2_model.mat', root, cfg);
