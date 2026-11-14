%% RESNET-18: res5a + res5b + classification head, controlled fine-tuning.
% Run setup_project first, or run this file from the project root.
clc; clear; close all;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));
[net, inputSize, cfg, experiment] = prepareResnet18FineTuning(root, "partial");
runTrainingFineTuning(net, inputSize, cfg.learnRate.transfer, ...
    'resnet18_partial_ft.mat', root, cfg, experiment);
