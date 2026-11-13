function [net, trainInfo] = runTraining(net, inputSize, learnRate, modelFile, root, cfg)
% RUNTRAINING Entraine un reseau avec le protocole commun (trainingConfig)
% et le sauvegarde dans models/<modelFile> au format attendu par evaluation/.
%
%   runTraining(net, [224 224 3], cfg.learnRate.transfer, 'resnet18_model.mat', root, cfg)
%
% Identique pour les trois modeles : split, augmentation, poids de classes,
% epochs, optimiseur, choix du meilleur epoch, format de sauvegarde.
% Le batch est propre a chaque modele (cfg.miniBatchSize.<modele>).

[dsTrain, dsVal, imdsTrain] = makeDatastores(root, inputSize, cfg);

% Nom court du modele (lenet5, mobilenetv2, resnet18) et son batch
[~, modelName] = fileparts(modelFile);
modelName = erase(string(modelName), "_model");
miniBatchSize = cfg.miniBatchSize.(char(modelName));

% --- Loss : entropie croisee, ponderee par classe si demande ---
if cfg.useClassWeights
    [w, classNames] = classWeights(imdsTrain.Labels);
    lossFcn = @(Y, T) crossentropy(Y, T, w, WeightsFormat="C");
    fprintf('Poids de classes : %s\n', mat2str(round(w, 2)));
else
    classNames = categories(imdsTrain.Labels);
    lossFcn = "crossentropy";
end

itersPerEpoch = floor(numel(imdsTrain.Files) / miniBatchSize);

% --- Courbes d'entrainement : meme figure pour les trois modeles ---
[progressFcn, progressFig] = makeCourseProgressOutputFcn( ...
    numel(imdsTrain.Files), miniBatchSize, cfg.maxEpochs, ...
    cfg.plotSmoothingWindow, modelName);

options = trainingOptions(cfg.optimizer, ...
    InitialLearnRate     = learnRate, ...
    MaxEpochs            = cfg.maxEpochs, ...
    MiniBatchSize        = miniBatchSize, ...
    Shuffle              = "every-epoch", ...
    ValidationData       = dsVal, ...
    ValidationFrequency  = itersPerEpoch, ...   % une validation par epoch
    OutputNetwork        = cfg.outputNetwork, ...
    Metrics              = "accuracy", ...
    ExecutionEnvironment = cfg.executionEnvironment, ...
    Plots                = "none", ...           % remplace par la figure ci-dessus
    OutputFcn            = progressFcn, ...
    Verbose              = true);

fprintf('Entrainement de %s : %d epochs, batch %d, lr %g\n', ...
    modelFile, cfg.maxEpochs, miniBatchSize, learnRate);
tStart = tic;
[net, trainInfo] = trainnet(dsTrain, net, lossFcn, options);
trainingTime = toc(tStart);
fprintf('Termine en %.1f min.\n', trainingTime / 60);

% --- Export de la figure de progression (pour le rapport) ---
figDir = fullfile(root, 'results', 'training');
if ~isfolder(figDir), mkdir(figDir); end
if isgraphics(progressFig)
    exportgraphics(progressFig, fullfile(figDir, modelName + "_progress.png"), Resolution=160);
    fprintf('Courbes : results/training/%s_progress.png\n', modelName);
end

% --- Sauvegarde : tout ce dont evaluation/ a besoin ---
modelsDir = fullfile(root, 'models');
if ~isfolder(modelsDir), mkdir(modelsDir); end
inputSize = inputSize(:)';
save(fullfile(modelsDir, modelFile), ...
    'net', 'trainInfo', 'classNames', 'inputSize', 'trainingTime', 'cfg', '-v7.3');
fprintf('Modele sauvegarde : models/%s\n', modelFile);
end
