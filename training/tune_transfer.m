function results = tune_transfer(modelName, opts)
% TUNE_TRANSFER Recherche des meilleurs hyperparametres pour un modele en
% transfer learning (MobileNetV2 ou ResNet-18), puis entrainement final.
%
%   tune_transfer("mobilenetv2")
%   tune_transfer("resnet18")
%   tune_transfer("mobilenetv2", QuickTest=true)   % test rapide du code
%
% Methode (backbone gele, cf. cfg.freezeBackbone) :
%   1. Les images passent UNE fois dans le backbone pre-entraine ; on garde
%      ses caracteristiques (sortie du global average pooling). Le train est
%      extrait sans augmentation + NumAugCopies copies augmentees (meme
%      augmentation que P1). Resultat mis en cache dans results/tuning/.
%   2. Recherche en 3 etapes sur la couche de classification, a partir de
%      ces caracteristiques (rapide, meme sur CPU) :
%        etape 1 : learning rate x taille de batch
%        etape 2 : regularisation L2 x dropout
%        etape 3 : poids de classes x augmentation
%      Critere de choix : macro-F1 sur la validation.
%      La MEME grille est utilisee pour les deux modeles.
%   3. La meilleure couche est reinseree dans le reseau complet, sauvegardee
%      dans models/<modele>_model.mat au format attendu par evaluation/.
%
% Sorties : results/tuning/<modele>_search.csv / .png,
%           results/training/<modele>_progress.png, models/<modele>_model.mat

arguments
    modelName (1,1) string {mustBeMember(modelName, ["mobilenetv2","resnet18"])}
    opts.QuickTest (1,1) logical = false
    opts.NumAugCopies (1,1) double {mustBeInteger, mustBeNonnegative} = 2
    opts.MaxEpochs (1,1) double {mustBeInteger, mustBePositive} = 12
end

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));
cfg = trainingConfig();
outDir = fullfile(root, 'results', 'tuning');
if ~isfolder(outDir), mkdir(outDir); end
tag = modelName;
if opts.QuickTest
    tag = modelName + "_quicktest";
    opts.MaxEpochs = 2;
end
tTotal = tic;

%% 1. RESEAU PRE-ENTRAINE ET COUCHE DE CARACTERISTIQUES
if opts.QuickTest
    net = imagePretrainedNetwork(modelName, NumClasses=11, Weights="none");
    if ~net.Initialized, net = initialize(net); end
else
    net = imagePretrainedNetwork(modelName, NumClasses=11);
end
inputSize = net.Layers(1).InputSize;
isFC = arrayfun(@(l) isa(l, 'nnet.cnn.layer.FullyConnectedLayer'), net.Layers);
iFC = find(isFC, 1, 'last');
fcName = string(net.Layers(iFC).Name);
featName = string(net.Layers(iFC-1).Name);
fprintf('%s : entree %s, caracteristiques = sortie de "%s", classifieur = "%s"\n', ...
    modelName, mat2str(inputSize), featName, fcName);

%% 2. EXTRACTION DES CARACTERISTIQUES (une seule fois, mise en cache)
cacheFile = fullfile(outDir, "features_" + tag + ".mat");
if isfile(cacheFile)
    fprintf('Caracteristiques lues dans le cache : %s\n', cacheFile);
    load(cacheFile, 'Xtr', 'Ytr', 'Xval', 'Yval', 'Xaug', 'Yaug');
else
    trainDir = fullfile(root, 'data', 'train');
    imdsTrain = load_split(fullfile(root, 'data', 'splits', 'train_files.csv'), trainDir);
    imdsVal   = load_split(fullfile(root, 'data', 'splits', 'validation_files.csv'), trainDir);
    if opts.QuickTest
        imdsTrain = splitEachLabel(imdsTrain, 8, 'randomized');
        imdsVal   = splitEachLabel(imdsVal, 4, 'randomized');
    end
    sz = inputSize(1:2);
    dsTrain = augmentedImageDatastore(sz, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
    dsVal   = augmentedImageDatastore(sz, imdsVal,   'ColorPreprocessing', 'gray2rgb');

    t = tic;
    fprintf('Extraction train (%d images)...\n', numel(imdsTrain.Files));
    Xtr = extractFeatures(net, dsTrain, featName, numel(imdsTrain.Files));
    Ytr = imdsTrain.Labels;
    fprintf('Extraction validation (%d images)...\n', numel(imdsVal.Files));
    Xval = extractFeatures(net, dsVal, featName, numel(imdsVal.Files));
    Yval = imdsVal.Labels;

    Xaug = zeros(0, size(Xtr, 2), 'single');
    Yaug = Ytr([]);   % categorical vide, memes categories
    rng(cfg.seed);
    for k = 1:opts.NumAugCopies
        fprintf('Extraction train augmente, copie %d/%d...\n', k, opts.NumAugCopies);
        dsAug = augmentedImageDatastore(sz, imdsTrain, ...
            'DataAugmentation', makeAugmenter(cfg), 'ColorPreprocessing', 'gray2rgb');
        range = cfg.aug.brightness;
        dsAug = transform(dsAug, @(d) brightnessBatch(d, range));
        Xaug = [Xaug; extractFeatures(net, dsAug, featName, numel(imdsTrain.Files))]; %#ok<AGROW>
        Yaug = [Yaug; Ytr]; %#ok<AGROW>
    end
    fprintf('Extraction terminee en %.1f min.\n', toc(t) / 60);
    save(cacheFile, 'Xtr', 'Ytr', 'Xval', 'Yval', 'Xaug', 'Yaug', '-v7.3');
end
classNames = categories(Ytr);
data = struct('Xtr', Xtr, 'Ytr', Ytr, 'Xval', Xval, 'Yval', Yval, ...
    'Xaug', Xaug, 'Yaug', Yaug, 'classNames', {classNames});

%% 3. RECHERCHE DES HYPERPARAMETRES (meme grille pour les deux modeles)
base = struct('LearnRate', 1e-3, 'MiniBatchSize', 32, 'L2', 1e-4, ...
    'Dropout', 0, 'ClassWeights', true, 'Augmentation', opts.NumAugCopies > 0);
grid1 = combos('LearnRate', [1e-4 3e-4 1e-3 3e-3], 'MiniBatchSize', [8 16 32 64]);
grid2 = combos('L2', [1e-5 1e-4 1e-3 1e-2], 'Dropout', [0 0.2 0.5]);
if opts.NumAugCopies > 0
    grid3 = combos('ClassWeights', [true false], 'Augmentation', [true false]);
else
    grid3 = combos('ClassWeights', [true false]);
end

rows = {};
best = base;
stages = {grid1, grid2, grid3};
for s = 1:numel(stages)
    grid = stages{s};
    stageScores = zeros(numel(grid), 1);
    for g = 1:numel(grid)
        p = best;
        f = fieldnames(grid{g});
        for j = 1:numel(f), p.(f{j}) = grid{g}.(f{j}); end
        t = tic;
        [~, m] = trainHead(p, data, opts.MaxEpochs, cfg.seed, []);
        sec = toc(t);
        stageScores(g) = m.MacroF1;
        rows(end+1, :) = {s, p.LearnRate, p.MiniBatchSize, p.L2, p.Dropout, ...
            p.ClassWeights, p.Augmentation, m.Accuracy, m.MacroF1, sec}; %#ok<AGROW>
        fprintf(['[etape %d | %2d/%2d] lr=%-7g batch=%-3d L2=%-7g drop=%-4g ' ...
            'poids=%d aug=%d -> acc %.3f | macro-F1 %.4f (%.0fs)\n'], ...
            s, g, numel(grid), p.LearnRate, p.MiniBatchSize, p.L2, p.Dropout, ...
            p.ClassWeights, p.Augmentation, m.Accuracy, m.MacroF1, sec);
    end
    [~, iBest] = max(stageScores);
    f = fieldnames(grid{iBest});
    for j = 1:numel(f), best.(f{j}) = grid{iBest}.(f{j}); end
end

results = cell2table(rows, 'VariableNames', {'Etape','LearnRate','MiniBatchSize', ...
    'L2','Dropout','ClassWeights','Augmentation','ValAccuracy','ValMacroF1','Secondes'});
writetable(results, fullfile(outDir, tag + "_search.csv"));
plotSearch(results, modelName, fullfile(outDir, tag + "_search.png"));
fprintf('\nMEILLEURS PARAMETRES %s : lr=%g, batch=%d, L2=%g, dropout=%g, poids=%d, aug=%d\n', ...
    modelName, best.LearnRate, best.MiniBatchSize, best.L2, best.Dropout, ...
    best.ClassWeights, best.Augmentation);

%% 4. ENTRAINEMENT FINAL AVEC LES MEILLEURS PARAMETRES (+ courbes)
nTrain = size(data.Xtr, 1) * (1 + best.Augmentation * opts.NumAugCopies);
[progressFcn, progressFig] = makeCourseProgressOutputFcn(nTrain, ...
    best.MiniBatchSize, opts.MaxEpochs, cfg.plotSmoothingWindow, modelName);
[head, m, trainInfo] = trainHead(best, data, opts.MaxEpochs, cfg.seed, progressFcn);
fprintf('Modele final : accuracy %.3f | macro-F1 %.4f (validation)\n', m.Accuracy, m.MacroF1);

% Reinsertion de la couche entrainee dans le reseau complet
L = net.Learnables;
H = head.Learnables;
for prm = ["Weights", "Bias"]
    L.Value{L.Layer == fcName & L.Parameter == prm} = ...
        H.Value{H.Layer == "fc" & H.Parameter == prm};
end
net.Learnables = L;

% Controle : le reseau complet doit predire comme la couche seule
nCheck = min(40, numel(data.Yval));
trainDir = fullfile(root, 'data', 'train');
imdsVal = load_split(fullfile(root, 'data', 'splits', 'validation_files.csv'), trainDir);
if ~opts.QuickTest
    dsCheck = augmentedImageDatastore(inputSize(1:2), subset(imdsVal, 1:nCheck), ...
        'ColorPreprocessing', 'gray2rgb');
    sFull = minibatchpredict(net, dsCheck);
    sHead = minibatchpredict(head, data.Xval(1:nCheck, :));
    [~, a] = max(sFull, [], 2); [~, b] = max(sHead, [], 2);
    fprintf('Controle reseau complet / couche seule : %d/%d predictions identiques\n', ...
        nnz(a == b), nCheck);
end

% Sauvegarde au format attendu par evaluation/
trainingTime = toc(tTotal);
inputSize = inputSize(:)';
bestParams = best;
searchTable = results;
if opts.QuickTest
    modelPath = fullfile(outDir, tag + "_model.mat");   % ne pas ecraser models/
else
    modelPath = fullfile(root, 'models', modelName + "_model.mat");
end
save(modelPath, 'net', 'trainInfo', 'classNames', 'inputSize', 'trainingTime', ...
    'cfg', 'bestParams', 'searchTable', '-v7.3');
fprintf('Modele sauvegarde : %s\n', modelPath);

figDir = fullfile(root, 'results', 'training');
if ~isfolder(figDir), mkdir(figDir); end
if isgraphics(progressFig)
    exportgraphics(progressFig, fullfile(figDir, tag + "_progress.png"), Resolution=160);
    close(progressFig);
end
fprintf('Termine en %.1f min.\n', trainingTime / 60);
end

%% ===================== FONCTIONS LOCALES =====================
function X = extractFeatures(net, ds, layerName, nObs)
F = minibatchpredict(net, ds, Outputs=layerName, MiniBatchSize=32);
if isa(F, 'dlarray'), F = extractdata(F); end
F = squeeze(F);
if size(F, 1) ~= nObs || (size(F, 2) == nObs && size(F, 1) ~= nObs)
    F = F.';
end
X = single(F);   % nObs x nFeatures
end

function data = brightnessBatch(data, range)
for i = 1:height(data)
    data.input{i} = randomBrightness(data.input{i}, range);
end
end

function [head, m, info] = trainHead(p, data, maxEpochs, seed, outputFcn)
rng(seed);
X = data.Xtr; Y = data.Ytr;
if p.Augmentation
    X = [X; data.Xaug]; Y = [Y; data.Yaug];
end
nFeat = size(X, 2);
layers = featureInputLayer(nFeat, Normalization="none");
if p.Dropout > 0, layers = [layers; dropoutLayer(p.Dropout)]; end
layers = [layers; fullyConnectedLayer(numel(data.classNames), Name="fc"); softmaxLayer];

if p.ClassWeights
    w = classWeights(Y);
    lossFcn = @(Yp, T) crossentropy(Yp, T, w, WeightsFormat="C");
else
    lossFcn = "crossentropy";
end
args = {"adam", ...
    "InitialLearnRate", p.LearnRate, "MaxEpochs", maxEpochs, ...
    "MiniBatchSize", p.MiniBatchSize, "L2Regularization", p.L2, ...
    "Shuffle", "every-epoch", "ValidationData", {data.Xval, data.Yval}, ...
    "ValidationFrequency", max(1, floor(size(X, 1) / p.MiniBatchSize)), ...
    "OutputNetwork", "best-validation", "Metrics", "accuracy", ...
    "ExecutionEnvironment", "cpu", "Verbose", false, "Plots", "none"};
if ~isempty(outputFcn), args = [args, {"OutputFcn", outputFcn}]; end
[head, info] = trainnet(X, Y, layers, lossFcn, trainingOptions(args{:}));

scores = minibatchpredict(head, data.Xval);
[~, idx] = max(scores, [], 2);
yPred = categorical(data.classNames(idx), data.classNames);
m = scoreMetrics(data.Yval, yPred, data.classNames);
end

function m = scoreMetrics(yTrue, yPred, classNames)
K = numel(classNames);
f1 = zeros(K, 1);
for k = 1:K
    c = classNames{k};
    tp = nnz(yTrue == c & yPred == c);
    prec = tp / max(nnz(yPred == c), 1);
    rec  = tp / max(nnz(yTrue == c), 1);
    f1(k) = 2 * prec * rec / max(prec + rec, eps);
end
m.MacroF1 = mean(f1);
m.Accuracy = mean(yTrue == yPred);
end

function grid = combos(name1, values1, name2, values2)
grid = {};
if nargin < 3
    for a = values1, grid{end+1} = struct(name1, a); end %#ok<AGROW>
    return
end
for a = values1
    for b = values2
        grid{end+1} = struct(name1, a, name2, b); %#ok<AGROW>
    end
end
end

function plotSearch(T, modelName, outFile)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [40 40 1600 520]);
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

S = T(T.Etape == 1, :);
ax = nexttile(tl);
heat(ax, S, 'LearnRate', 'MiniBatchSize', 'Learning rate', 'Taille de batch');
title(ax, 'Étape 1 : learning rate × batch');

S = T(T.Etape == 2, :);
ax = nexttile(tl);
heat(ax, S, 'L2', 'Dropout', 'Régularisation L2', 'Dropout');
title(ax, 'Étape 2 : L2 × dropout');

S = T(T.Etape == 3, :);
ax = nexttile(tl);
labels = strings(height(S), 1);
for i = 1:height(S)
    labels(i) = sprintf('poids %s / aug %s', onoff(S.ClassWeights(i)), onoff(S.Augmentation(i)));
end
bar(ax, categorical(labels, labels), S.ValMacroF1);
ylabel(ax, 'Macro-F1 (validation)'); grid(ax, 'on');
lo = min(S.ValMacroF1); hi = max(S.ValMacroF1);
ylim(ax, [max(0, lo - 0.05) min(1, hi + 0.02)]);
title(ax, 'Étape 3 : poids de classes × augmentation');

title(tl, sprintf('%s : recherche des hyperparamètres (macro-F1 sur la validation)', modelName));
exportgraphics(fig, outFile, 'Resolution', 160);
close(fig);
end

function heat(ax, S, xName, yName, xLabel, yLabel)
xs = unique(S.(xName)); ys = unique(S.(yName));
M = nan(numel(ys), numel(xs));
for i = 1:height(S)
    M(ys == S.(yName)(i), xs == S.(xName)(i)) = S.ValMacroF1(i);
end
imagesc(ax, M); colormap(ax, parula); colorbar(ax);
xticks(ax, 1:numel(xs)); xticklabels(ax, compose('%g', xs));
yticks(ax, 1:numel(ys)); yticklabels(ax, compose('%g', ys));
xlabel(ax, xLabel); ylabel(ax, yLabel);
for i = 1:numel(ys)
    for j = 1:numel(xs)
        if ~isnan(M(i, j))
            text(ax, j, i, sprintf('%.3f', M(i, j)), 'HorizontalAlignment', 'center', ...
                'Color', 'w', 'FontWeight', 'bold');
        end
    end
end
end

function s = onoff(b)
if b, s = "oui"; else, s = "non"; end
end
