function [net, inputSize, cfg, experiment] = prepareResnet18FineTuning(root, mode)
% Prepare partial or full fine-tuning from the same saved ResNet-18 baseline.
mode = string(mode);
assert(ismember(mode, ["partial", "full"]), 'Use partial or full fine-tuning.');

% Shared settings. Data augmentation and class-weighted loss stay unchanged.
cfg = trainingConfig();
cfg.seed = 42;
cfg.optimizer = "adam";
cfg.learnRate.transfer = 1e-5;
cfg.maxEpochs = 10;
cfg.miniBatchSize.resnet18 = 8;
cfg.modelKey = "resnet18";
cfg.executionEnvironment = "gpu";
cfg.validationFrequency = 200;
cfg.outputNetwork = "best-validation";
cfg.objectiveMetricName = "loss";
cfg.batchNormalizationStatistics = "moving";
cfg.resetInputNormalization = false;
cfg.freezeBackbone = false; % The per-parameter factors below control freezing.
cfg.preventOverwrite = true;
cfg.evaluateTrainingSet = true;
cfg.saveFinalCheckpoint = true;
rng(cfg.seed, 'twister');

% Both experiments load the baseline independently; neither loads the other.
source = fullfile(root, 'models', 'resnet18_model.mat');
assert(isfile(source), 'Baseline missing: models/resnet18_model.mat');
S = load(source);
if isfield(S, 'net')
    net = S.net;
elseif isfield(S, 'netResNetTrained')
    net = S.netResNetTrained; % Variable name in the existing team checkpoint.
else
    error('Baseline must contain net or netResNetTrained.');
end
assert(isa(net, 'dlnetwork') && net.Initialized, ...
    'Baseline must contain an initialized dlnetwork.');
inputSize = net.Layers(1).InputSize;

% ResNet-18 has a single fully connected classification head. Find it by
% layer type rather than assuming a layer name or learnables-table position.
headNames = strings(0,1);
for k = 1:numel(net.Layers)
    layer = net.Layers(k);
    if isa(layer, 'nnet.cnn.layer.FullyConnectedLayer')
        headNames(end+1,1) = string(layer.Name); %#ok<AGROW>
        assert(layer.OutputSize == 11, 'Expected an 11-class classification head.');
    end
end
assert(isscalar(headNames), 'Expected one fully connected ResNet-18 head.');

% Keep output classes aligned with trainnet's categorical-label ordering.
labels = load_split(fullfile(root, 'data', 'splits', 'train_files.csv'), ...
    fullfile(root, 'data', 'train'));
assert(isfield(S, 'classNames') && ...
    isequal(string(S.classNames(:)), string(categories(labels.Labels))), ...
    'Baseline classNames must match the training split class order.');

L = net.Learnables;
names = string(L.Layer);
% BN scale and offset use bn5a/b prefixes, unlike the convolution names.
lastBlocks = startsWith(names, ["res5a_", "res5b_", "bn5a_", "bn5b_"]);
assert(any(startsWith(names,"res5a_")) && any(startsWith(names,"res5b_")) && ...
    any(startsWith(names,"bn5a_")) && any(startsWith(names,"bn5b_")), ...
    'Expected res5a/res5b convolution and Batch Normalization parameters.');
trainable = lastBlocks | names == headNames;
if mode == "full", trainable(:) = true; end

% Change only learning-rate factors: preserve weights, BN state and graph.
for k = 1:height(L)
    factor = double(trainable(k));
    net = setLearnRateFactor(net, L.Layer(k), L.Parameter(k), factor);
    assert(getLearnRateFactor(net, L.Layer(k), L.Parameter(k)) == factor, ...
        'Incorrect learning-rate factor for %s/%s.', L.Layer(k), L.Parameter(k));
end
counts = cellfun(@numel, L.Value);
summary = table(names, string(L.Parameter), double(trainable), counts, ...
    'VariableNames', {'Layer','Parameter','LearnRateFactor','ParameterCount'});
experiment.name = "resnet18_" + mode + "_ft";
experiment.depth = mode;
experiment.initializationCheckpoint = "models/resnet18_model.mat";
experiment.learnables = summary;
experiment.trainableParameterCount = sum(counts(trainable));
% Frozen BN learnables do not freeze running statistics. Both arms use the
% same moving-statistics policy in all BN layers.
disp(summary(trainable,:));
fprintf('%s: %d trainable scalar parameters.\n', ...
    experiment.name, experiment.trainableParameterCount);
end
