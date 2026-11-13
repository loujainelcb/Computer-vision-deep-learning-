function [net, trainInfo] = runTraining(net, inputSize, learnRate, modelFile, root, cfg)
% RUNTRAINING Entraine un reseau avec le protocole commun (trainingConfig)
% et le sauvegarde dans models/<modelFile> au format attendu par evaluation/.
%
%   runTraining(net, [224 224 3], cfg.learnRate.transfer, 'resnet18_model.mat', root, cfg)
%
% Identique pour les trois modeles : split, augmentation, poids de classes,
% epochs, batch, optimiseur, choix du meilleur epoch, format de sauvegarde.

[dsTrain, dsVal, imdsTrain] = makeDatastores(root, inputSize, cfg);

% --- Loss : entropie croisee, ponderee par classe si demande ---
if cfg.useClassWeights
    [w, classNames] = classWeights(imdsTrain.Labels);
    lossFcn = @(Y, T) crossentropy(Y, T, w, WeightsFormat="C");
    fprintf('Poids de classes : %s\n', mat2str(round(w, 2)));
else
    classNames = categories(imdsTrain.Labels);
    lossFcn = "crossentropy";
end

itersPerEpoch = floor(numel(imdsTrain.Files) / cfg.miniBatchSize);
options = trainingOptions(cfg.optimizer, ...
    InitialLearnRate     = learnRate, ...
    MaxEpochs            = cfg.maxEpochs, ...
    MiniBatchSize        = cfg.miniBatchSize, ...
    Shuffle              = "every-epoch", ...
    ValidationData       = dsVal, ...
    ValidationFrequency  = itersPerEpoch, ...   % une validation par epoch
    OutputNetwork        = cfg.outputNetwork, ...
    Metrics              = "accuracy", ...
    ExecutionEnvironment = cfg.executionEnvironment, ...
    Plots                = "training-progress", ...
    Verbose              = true);

fprintf('Entrainement de %s : %d epochs, batch %d, lr %g\n', ...
    modelFile, cfg.maxEpochs, cfg.miniBatchSize, learnRate);
tStart = tic;
[net, trainInfo] = trainnet(dsTrain, net, lossFcn, options);
trainingTime = toc(tStart);
fprintf('Termine en %.1f min.\n', trainingTime / 60);

% --- Sauvegarde : tout ce dont evaluation/ a besoin ---
modelsDir = fullfile(root, 'models');
if ~isfolder(modelsDir), mkdir(modelsDir); end
inputSize = inputSize(:)';
save(fullfile(modelsDir, modelFile), ...
    'net', 'trainInfo', 'classNames', 'inputSize', 'trainingTime', 'cfg', '-v7.3');
fprintf('Modele sauvegarde : models/%s\n', modelFile);
end
