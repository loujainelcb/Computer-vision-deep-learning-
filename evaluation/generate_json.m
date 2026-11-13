%% P4 - JSON OFFICIEL FOOD-11 (PREDICTIONS SUR LE TEST)
% Executer APRES evaluate_models.m puis compare_models.m.
%
% Aucun label de data/test n'est connu : PAS de calcul de F1 sur le test.
% Chaque entree : "nom_image_sans_extension": "nom_de_classe_en_minuscules"
% Exemple : "0": "bread"
%
% La mise a jour du leaderboard se fait seulement pour le resultat final.

clearvars; clc;

%% A. PARAMETRES
selectedModel="auto";             % ou "LeNet-5", "MobileNetV2", "ResNet"
requireAllThreeModels=true;       % securite : ne pas finaliser trop tot
miniBatchSize=32;                 % sans effet sur les predictions
executionEnvironment='auto';     % 'cpu' si GPU indisponible
expectedTestCount=3097;
allowedClasses=["bread","dairy","dessert","egg","fried","meat", ...
    "noodles-pasta","rice","seafood","soup","vegetable-fruit"];
plannedModels=["LeNet-5","MobileNetV2","ResNet"];

%% B. TROUVER LE PROJET ET LE TABLEAU DE VALIDATION
root=locateProjectRoot(fileparts(mfilename('fullpath')));
resultDir=fullfile(root,'results','evaluation');
metricsPath=fullfile(resultDir,'model_metrics.csv');
testDir=fullfile(root,'data','test');
if ~isfile(metricsPath)
    error('Lancer d''abord evaluate_models.m (model_metrics.csv absent).');
end
if ~isfolder(testDir)
    error('Dossier de test absent : remplir data/test/ avec les 3097 images.');
end
T=readtable(metricsPath,'TextType','string');
if isempty(T),error('Aucun modele evalue sur validation.');end
if requireAllThreeModels && ~all(ismember(plannedModels,string(T.Model)))
    error(['Les trois modeles ne sont pas encore evalues. ' ...
           'Ne pas finaliser le JSON trop tot. ' ...
           'Pour un ESSAI PROVISOIRE, mettre requireAllThreeModels=false.']);
end
T=sortrows(T,'FrugalityScore','descend');
if strcmpi(selectedModel,"auto")
    k=1;
else
    k=find(strcmpi(string(T.Model),selectedModel),1);
    if isempty(k),error('Modele choisi absent des evaluations.');end
end
modelName=string(T.Model(k));
modelFile=string(T.ModelFile(k));
[~,basename,ext]=fileparts(modelFile);
modelPath=fullfile(root,'models',char(string(basename)+string(ext)));
if ~isfile(modelPath)
    error('Modele entraine introuvable : %s',modelPath);
end
fprintf('Modele retenu : %s | Validation macro-F1 %.4f | %.3f M params\n', ...
    modelName,T.MacroF1(k),T.ParametersM(k));

%% C. LIRE LES IMAGES TEST SANS AUCUNE AUGMENTATION
imds=imageDatastore(testDir,'IncludeSubfolders',true, ...
    'FileExtensions',{'.jpg','.jpeg','.png'});
paths=string(imds.Files);
if numel(paths)~=expectedTestCount
    error('Le dossier test contient %d images (attendu : %d).', ...
        numel(paths),expectedTestCount);
end
ids=strings(numel(paths),1);
for i=1:numel(paths)
    [~,stem,~]=fileparts(paths(i));
    ids(i)=string(stem);
end
if numel(unique(ids))~=numel(ids)
    error('Deux images test ont le meme nom sans extension.');
end
% Tri numerique pour les noms 0.jpg, 1.jpg, 2.jpg, ...
values=str2double(ids);
if all(isfinite(values))
    [~,order]=sort(values);
else
    [~,order]=sort(ids);
end
ids=ids(order);
paths=paths(order);
imds=imageDatastore(cellstr(paths));

%% D. PREDIRE AVEC LE RESEAU DEJA ENTRAINE
[net,classOrder,inputSize]=loadClassificationModel(modelPath);
ds=augmentedImageDatastore(inputSize(1:2),imds, ...
    'ColorPreprocessing','gray2rgb');
yPred=predictLabels(net,ds,classOrder,miniBatchSize,executionEnvironment);
yPred=lower(strtrim(string(yPred(:))));
if numel(yPred)~=expectedTestCount
    error('Il manque des predictions (%d/%d).',numel(yPred),expectedTestCount);
end
if any(~ismember(yPred,allowedClasses))
    error('Au moins une prediction a une classe inconnue.');
end

%% E. CONSTRUIRE LE JSON SANS ALTERER LES CLES NUMERIQUES
% IMPORTANT : une struct MATLAB avec le champ "0" n'est pas valide.
% On ecrit donc les cles JSON directement, avec jsonencode pour les textes.
if ~isfolder(resultDir),mkdir(resultDir);end
outputJson=fullfile(resultDir,'submission_food11.json');
fid=fopen(outputJson,'w','n','UTF-8');
if fid<0,error('Creation du JSON impossible : %s',outputJson);end
try
    fprintf(fid,'{\n');
    for i=1:numel(ids)
        if i<numel(ids),suffix=',';else,suffix='';end
        fprintf(fid,'  %s: %s%s\n', ...
            jsonencode(char(ids(i))),jsonencode(char(yPred(i))),suffix);
    end
    fprintf(fid,'}\n');
    fclose(fid);
catch ME
    fclose(fid);
    rethrow(ME);
end

% Verification de syntaxe JSON. MATLAB peut renommer les champs commencant
% par un chiffre lorsqu'il lit le fichier : cela ne modifie PAS le JSON ecrit.
parsed=jsondecode(fileread(outputJson));
if numel(fieldnames(parsed))~=expectedTestCount
    error('Verification JSON echouee : nombre de cles incorrect.');
end

%% F. FICHIERS DE CONTROLE (NE PAS SOUMETTRE AU SERVEUR)
Tpred=table(ids,yPred,'VariableNames',{'ImageID','PredictedClass'});
writetable(Tpred,fullfile(resultDir,'test_predictions.csv'));
counts=zeros(numel(allowedClasses),1);
for i=1:numel(allowedClasses),counts(i)=sum(yPred==allowedClasses(i));end
distribution=table(allowedClasses(:),counts, ...
    'VariableNames',{'Class','PredictedImages'});
writetable(distribution,fullfile(resultDir,'test_class_distribution.csv'));
try
    fig=figure('Visible','off','Position',[50 50 1150 570]);
    bar(categorical(cellstr(allowedClasses)),counts);
    ylabel('Nombre d''images predites');grid on;
    title(char("Distribution des predictions test - "+modelName));
    exportgraphics(fig,fullfile(resultDir,'test_class_distribution.png'), ...
        'Resolution',160);
    close(fig);
catch plotErr
    warning('Graphique de repartition non cree : %s',plotErr.message);
    if exist('fig','var') && isgraphics(fig),close(fig);end
end

fprintf('\n=== SOUMISSION PRETE ===\n');
fprintf('Fichier JSON : %s\n',outputJson);
fprintf('Images : %d | Cles distinctes : %d\n',numel(ids),numel(unique(ids)));
fprintf('Champ "Model parameter count (Millions)" sur le site : %.6f\n', ...
    T.ParametersM(k));
fprintf('Ne cochez "Update leaderboard" qu''au moment du resultat final.\n');

%% FONCTIONS LOCALES (dupliquees pour garder exactement 3 scripts autonomes)
function root=locateProjectRoot(here)
    dirs={here,fileparts(here),fileparts(fileparts(here))};
    for j=1:numel(dirs)
        if isfolder(fullfile(dirs{j},'models')) && ...
                isfolder(fullfile(dirs{j},'data'))
            root=dirs{j};return;
        end
    end
    error('Racine du projet Food-11 introuvable.');
end
function [net,order,inputSize]=loadClassificationModel(matPath)
    S=load(matPath);net=[];
    fields=[{'trainedNet','net','trainedNetwork','network'},fieldnames(S)'];
    for j=1:numel(fields)
        fn=fields{j};
        if ~isfield(S,fn),continue;end
        candidate=S.(fn);
        if isa(candidate,'SeriesNetwork') || isa(candidate,'DAGNetwork') || ...
                isa(candidate,'dlnetwork')
            net=candidate;break;
        end
    end
    if isempty(net),error('Le .mat ne contient pas de reseau MATLAB reconnu.');end
    order=strings(0,1);
    for field={'classNames','classes','classLabels','labelNames'}
        if isfield(S,field{1})
            order=string(S.(field{1}));order=order(:);break;
        end
    end
    if isa(net,'dlnetwork') && numel(order)~=11
        % trainnet + labels categorical => sorties dans l'ordre alphabetique
        % des classes. On l'utilise par defaut si classNames n'a pas ete sauve.
        order = ["bread";"dairy";"dessert";"egg";"fried";"meat"; ...
            "noodles-pasta";"rice";"seafood";"soup";"vegetable-fruit"];
        warning(['%s : classNames absent du .mat, ordre alphabetique ' ...
            'suppose (comportement par defaut de trainnet).'],matPath);
    end
    inputSize=[];
    for j=1:numel(net.Layers)
        if isprop(net.Layers(j),'InputSize')
            v=net.Layers(j).InputSize;
            if isnumeric(v) && numel(v)>=2,inputSize=double(v);break;end
        end
    end
    if isempty(inputSize) && isfield(S,'inputSize')
        inputSize=double(S.inputSize);
    end
    if numel(inputSize)<2
        error('Taille entree inconnue : sauvegarder inputSize dans .mat.');
    end
end
function y=predictLabels(net,ds,classOrder,batchSize,environment)
    if isa(net,'dlnetwork')
        scores=minibatchpredict(net,ds,'MiniBatchSize',batchSize, ...
            'ExecutionEnvironment',environment);
        if isa(scores,'dlarray'),scores=extractdata(scores);end
        if isa(scores,'gpuArray'),scores=gather(scores);end
        scores=squeeze(scores);
        if size(scores,2)==numel(classOrder)
            [~,idx]=max(scores,[],2);
        elseif size(scores,1)==numel(classOrder)
            [~,idx]=max(scores,[],1);
        else
            error('Scores incompatibles avec 11 classNames.');
        end
        y=classOrder(idx(:));
    else
        y=classify(net,ds,'MiniBatchSize',batchSize, ...
            'ExecutionEnvironment',environment);
    end
end
