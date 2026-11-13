%% Split stratifié train (80 %) / validation (20 %)
clear; clc;

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'src'));
trainDir  = fullfile(root, 'data', 'train');
splitsDir = fullfile(root, 'data', 'splits');
if ~exist(splitsDir, 'dir'), mkdir(splitsDir); end

imds = imageDatastore(trainDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% splitEachLabel coupe chaque classe à 80/20 -> les proportions sont conservées
rng(42);   % seed fixe pour avoir toujours le même split
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

fprintf('Train : %d images\n', numel(imdsTrain.Files));
fprintf('Validation : %d images\n', numel(imdsVal.Files));

% Vérification des proportions
cTrain = countEachLabel(imdsTrain);
cVal   = countEachLabel(imdsVal);
cTrain.Pct = 100 * cTrain.Count / sum(cTrain.Count);
cVal.Pct   = 100 * cVal.Count / sum(cVal.Count);
disp(table(cTrain.Label, cTrain.Pct, cVal.Pct, 'VariableNames', {'Classe', 'PctTrain', 'PctVal'}));

% Sauvegarde des listes de fichiers, en chemins RELATIFS a la racine
% (data/train/<classe>/<image>.jpg) pour que le CSV marche sur tous les PC.
writetable(table(relPaths(imdsTrain), imdsTrain.Labels, 'VariableNames', {'File', 'Label'}), ...
    fullfile(splitsDir, 'train_files.csv'));
writetable(table(relPaths(imdsVal), imdsVal.Labels, 'VariableNames', {'File', 'Label'}), ...
    fullfile(splitsDir, 'validation_files.csv'));

function p = relPaths(imds)
[~, nom, ext] = fileparts(string(imds.Files));
p = "data/train/" + string(imds.Labels) + "/" + nom + ext;
end
