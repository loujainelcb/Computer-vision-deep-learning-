function [dsTrain, dsVal, imdsTrain, imdsVal] = makeDatastores(root, inputSize, cfg)
% MAKEDATASTORES Datastores train / validation communs aux trois modeles.
%
%   [dsTrain, dsVal, imdsTrain, imdsVal] = makeDatastores(root, [224 224])
%
% - split : data/splits/*.csv (P1, seed 42), identique pour tous les modeles
% - train : redimensionnement + augmentation complete de P1 + luminosite
% - val   : redimensionnement UNIQUEMENT (meme traitement que evaluation/)
% Seule la taille d'entree depend du modele.
if nargin < 3, cfg = trainingConfig(); end
inputSize = inputSize(1:2);

trainDir  = fullfile(root, 'data', 'train');
splitsDir = fullfile(root, 'data', 'splits');
imdsTrain = load_split(fullfile(splitsDir, 'train_files.csv'), trainDir);
imdsVal   = load_split(fullfile(splitsDir, 'validation_files.csv'), trainDir);

dsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', makeAugmenter(cfg), ...
    'ColorPreprocessing', 'gray2rgb');
range = cfg.aug.brightness;
dsTrain = transform(dsTrain, @(data) brightnessBatch(data, range));

dsVal = augmentedImageDatastore(inputSize, imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

fprintf('Split : %d train / %d validation | entree %dx%d\n', ...
    numel(imdsTrain.Files), numel(imdsVal.Files), inputSize(1), inputSize(2));
end

function data = brightnessBatch(data, range)
for i = 1:height(data)
    data.input{i} = randomBrightness(data.input{i}, range);
end
end
