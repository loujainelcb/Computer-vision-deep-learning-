function [net, trainInfo] = runTrainingFineTuning(net, inputSize, learnRate, modelFile, root, cfg, experiment)
% RUNTRAININGFINETUNING Train and save the ResNet-18 fine-tuning experiments.
% Reuses the project datastores, weighted loss, progress plots and save format.
% Saves best-validation and final-epoch checkpoints when configured in cfg.

if nargin < 7, experiment = []; end
saveFinal = ~isempty(experiment) && isfield(cfg,'saveFinalCheckpoint') && cfg.saveFinalCheckpoint;
[~, baseName] = fileparts(modelFile);
finalModelFile = string(baseName) + "_final.mat";
if isfield(cfg,'preventOverwrite') && cfg.preventOverwrite
    assert(~isfile(fullfile(root,'models',modelFile)), 'Model output already exists.');
    if saveFinal
        assert(~isfile(fullfile(root,'models',finalModelFile)), 'Final model output already exists.');
    end
end
[dsTrain, dsVal, imdsTrain, imdsVal] = makeDatastores(root, inputSize, cfg);

% Nom court du modele (lenet5, mobilenetv2, resnet18) et son batch
[~, modelName] = fileparts(modelFile);
modelName = erase(string(modelName), "_model");
modelKey = modelName;
if isfield(cfg,'modelKey'), modelKey = cfg.modelKey; end
miniBatchSize = cfg.miniBatchSize.(char(modelKey));

% Keep each experiment's exports and temporary checkpoints together.
figDir = fullfile(root, 'results', 'training', 'finetuning', char(experiment.depth));
if ~isfolder(figDir), mkdir(figDir); end

% --- Loss : entropie croisee, ponderee par classe si demande ---
if cfg.useClassWeights
    [w, classNames] = classWeights(imdsTrain.Labels);
    lossFcn = @(Y, T) crossentropy(Y, T, w, WeightsFormat="C");
    fprintf('Poids de classes : %s\n', mat2str(round(w, 2)));
else
    w = [];
    classNames = categories(imdsTrain.Labels);
    lossFcn = "crossentropy";
end

itersPerEpoch = floor(numel(imdsTrain.Files) / miniBatchSize);
validationFrequency = itersPerEpoch;
if isfield(cfg,'validationFrequency'), validationFrequency = cfg.validationFrequency; end

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
    ValidationFrequency  = validationFrequency, ...
    OutputNetwork        = cfg.outputNetwork, ...
    Metrics              = "accuracy", ...
    ExecutionEnvironment = cfg.executionEnvironment, ...
    Plots                = "none", ...           % remplace par la figure ci-dessus
    OutputFcn            = progressFcn, ...
    Verbose              = true);
% Optional experiment settings; baseline options and defaults are unchanged.
if isfield(cfg,'objectiveMetricName'), options.ObjectiveMetricName = cfg.objectiveMetricName; end
if isfield(cfg,'batchNormalizationStatistics')
    options.BatchNormalizationStatistics = cfg.batchNormalizationStatistics;
end
if isfield(cfg,'resetInputNormalization')
    options.ResetInputNormalization = cfg.resetInputNormalization;
end
if saveFinal
    % trainnet returns the best network. Independently capture the CURRENT
    % network at the end of epoch 10 using MATLAB's built-in checkpoint saver.
    checkpointDir = tempname(figDir);
    mkdir(checkpointDir);
    options.CheckpointPath = checkpointDir;
    options.CheckpointFrequency = cfg.maxEpochs;
    options.CheckpointFrequencyUnit = 'epoch';
    options.ValidationPatience = Inf; % Always complete all 10 epochs.
end

fprintf('Entrainement de %s : %d epochs, batch %d, lr %g\n', ...
    modelFile, cfg.maxEpochs, miniBatchSize, learnRate);
tStart = tic;
if ~isempty(experiment), rng(cfg.seed,'twister'); end
[net, trainInfo] = trainnet(dsTrain, net, lossFcn, options);
trainingTime = toc(tStart);
fprintf('Termine en %.1f min.\n', trainingTime / 60);
if saveFinal
    history = trainInfo.TrainingHistory;
    finalIteration = history.Iteration(end);
    assert(trainInfo.StopReason == "Max epochs completed" && history.Epoch(end) == cfg.maxEpochs, ...
        'Training did not complete all epochs; refusing to label a checkpoint as final.');
    % Match the actual last iteration, not an earlier epoch or the best model.
    checkpoint = dir(fullfile(checkpointDir, "net_checkpoint__" + finalIteration + "__*.mat"));
    assert(isscalar(checkpoint), 'The last-iteration checkpoint was not saved.');
    rawFinalPath = fullfile(checkpoint(1).folder,checkpoint(1).name);
    finalSaved = load(rawFinalPath,'net');
    finalNet = finalSaved.net;
end

% Evaluate the actual selected network, using deterministic resize only.
if ~isempty(experiment)
    experiment = evaluateCheckpoint(net,experiment,inputSize,cfg,imdsTrain,imdsVal,classNames,w);
    experiment.checkpointType = "best-validation";
    history = trainInfo.ValidationHistory;
    experiment.bestObservedValidationAccuracy = max(history.Accuracy,[],'omitnan')/100;
    experiment.minimumValidationLoss = min(history.Loss,[],'omitnan');
    experiment.selectedCheckpointIteration = trainInfo.OutputNetworkIteration;
    th = trainInfo.TrainingHistory;
    selectedRow = find(th.Iteration==trainInfo.OutputNetworkIteration,1);
    experiment.selectedCheckpointEpoch = th.Epoch(selectedRow);
    experiment.trainingDurationSeconds = trainingTime;
    experiment.trainingStopReason = trainInfo.StopReason;
    if saveFinal
        finalExperiment = evaluateCheckpoint(finalNet,experiment,inputSize,cfg,imdsTrain,imdsVal,classNames,w);
        finalExperiment.checkpointType = "final";
        finalExperiment.selectedCheckpointIteration = finalIteration;
        finalExperiment.selectedCheckpointEpoch = cfg.maxEpochs;
    end
    fprintf('Saved validation accuracy %.2f%%; best observed %.2f%%; minimum loss %.6f; selected iteration %d\n', ...
        100*experiment.validation.accuracy,100*experiment.bestObservedValidationAccuracy, ...
        experiment.minimumValidationLoss,experiment.selectedCheckpointIteration);
end

% --- Export de la figure de progression (pour le rapport) ---
if isgraphics(progressFig)
    exportgraphics(progressFig, fullfile(figDir, modelName + "_progress.png"), Resolution=160);
    fprintf('Courbes : %s\n', fullfile(figDir, modelName + "_progress.png"));
end
if ~isempty(experiment)
    writetable(experiment.learnables,fullfile(figDir,modelName+"_learnables.csv"));
    writetable(trainInfo.TrainingHistory,fullfile(figDir,modelName+"_training_history.csv"));
    writetable(trainInfo.ValidationHistory,fullfile(figDir,modelName+"_validation_history.csv"));
end

% --- Sauvegarde : tout ce dont evaluation/ a besoin ---
modelsDir = fullfile(root, 'models');
if ~isfolder(modelsDir), mkdir(modelsDir); end
inputSize = inputSize(:)';
if isfield(cfg,'preventOverwrite') && cfg.preventOverwrite
    assert(~isfile(fullfile(modelsDir,modelFile)), 'Output appeared during training; refusing overwrite.');
end
save(fullfile(modelsDir, modelFile), ...
    'net', 'trainInfo', 'classNames', 'inputSize', 'trainingTime', 'cfg', '-v7.3');
if ~isempty(experiment)
    save(fullfile(modelsDir,modelFile),'experiment','-append');
end
fprintf('Modele sauvegarde : models/%s\n', modelFile);
if saveFinal
    assert(~isfile(fullfile(modelsDir,finalModelFile)), 'Final output already exists.');
    % Keep the original run's TrainingInfo in both files. Its OutputNetworkIteration
    % describes trainnet's BEST return; experiment identifies each saved checkpoint.
    finalSaved = struct('net',finalNet,'trainInfo',trainInfo,'classNames',{classNames}, ...
        'inputSize',inputSize,'trainingTime',trainingTime,'cfg',cfg,'experiment',finalExperiment);
    save(fullfile(modelsDir,finalModelFile),'-struct','finalSaved','-v7.3');
    fprintf('Final model: models/%s (iteration %d, epoch %d)\n', ...
        finalModelFile,finalIteration,cfg.maxEpochs);
    % Delete only this run's temporary raw checkpoint after successful export.
    delete(rawFinalPath);
    rmdir(checkpointDir);
end
end

function experiment = evaluateCheckpoint(net,experiment,inputSize,cfg,imdsTrain,imdsVal,classNames,w)
% Compare saved networks with deterministic preprocessing, never augmentation.
batchSize = cfg.miniBatchSize.(char(cfg.modelKey));
ds = augmentedImageDatastore(inputSize(1:2),imdsVal,ColorPreprocessing='gray2rgb');
scores = minibatchpredict(net,ds,MiniBatchSize=batchSize,ExecutionEnvironment=cfg.executionEnvironment);
experiment.validation = classificationScoreMetrics(scores,imdsVal.Labels,classNames,w);
if cfg.evaluateTrainingSet
    ds = augmentedImageDatastore(inputSize(1:2),imdsTrain,ColorPreprocessing='gray2rgb');
    scores = minibatchpredict(net,ds,MiniBatchSize=batchSize,ExecutionEnvironment=cfg.executionEnvironment);
    experiment.training = classificationScoreMetrics(scores,imdsTrain.Labels,classNames,w);
end
end
