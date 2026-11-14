%% RESNET-18: entire backbone + classification head, controlled fine-tuning.
% Always starts from the shared baseline, never from the partial result.
clc; clear; close all;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));
[net, inputSize, cfg, experiment] = prepareResnet18FineTuning(root, "full");
runTrainingFineTuning(net, inputSize, cfg.learnRate.transfer, ...
    'resnet18_full_ft.mat', root, cfg, experiment);
