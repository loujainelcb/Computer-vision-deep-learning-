%% SETUP DU PROJET FOOD-11
% A lancer une fois par session MATLAB, depuis n'importe ou :
%   >> setup_project
%
% - se place a la racine du projet
% - ajoute les dossiers de code au path MATLAB
% - cree les dossiers de sortie manquants
% - verifie que les donnees et les modeles sont presents

root = fileparts(mfilename('fullpath'));
cd(root);

addpath(fullfile(root, 'src'), ...
        fullfile(root, 'preprocessing'), ...
        fullfile(root, 'training'), ...
        fullfile(root, 'evaluation'));

dossiers = {fullfile('data','train'), fullfile('data','test'), fullfile('data','splits'), ...
            'models', fullfile('results','preprocessing'), fullfile('results','training'), ...
            fullfile('results','evaluation'), fullfile('results','presentation')};
for k = 1:numel(dossiers)
    if ~isfolder(dossiers{k}), mkdir(dossiers{k}); end
end

fprintf('Projet Food-11 : %s\n\n', root);

% --- Donnees ---
d = dir(fullfile('data','train'));
nClasses = nnz([d.isdir] & ~startsWith({d.name}, '.'));
nTest    = numel(dir(fullfile('data','test','*.jpg')));
fprintf('data/train  : %d dossiers de classes (attendu : 11)\n', nClasses);
fprintf('data/test   : %d images (attendu : 3097)\n', nTest);
fprintf('data/splits : train_files.csv %s | validation_files.csv %s\n', ...
    ok(isfile(fullfile('data','splits','train_files.csv'))), ...
    ok(isfile(fullfile('data','splits','validation_files.csv'))));

% --- Modeles ---
modeles = {'lenet5_model.mat','mobilenetv2_model.mat','resnet18_model.mat'};
fprintf('\nModeles entraines (models/) :\n');
for k = 1:numel(modeles)
    fprintf('  %-24s %s\n', modeles{k}, ok(isfile(fullfile('models', modeles{k}))));
end

if nClasses < 11 || nTest == 0
    fprintf(['\n>> Copier le dataset Food-11 dans data/train/<classe>/*.jpg ' ...
             'et data/test/*.jpg avant de lancer les scripts.\n']);
end
fprintf('\nOrdre : preprocessing/ -> training/ -> evaluation/ (voir README.md)\n');

function s = ok(b)
    if b, s = 'OK'; else, s = 'MANQUANT'; end
end
