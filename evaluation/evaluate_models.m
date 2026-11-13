%% P4 - EVALUATION DES RESEAUX SUR LA VALIDATION (Food-11)
% INSA Lyon - Challenge Food-11 | MATLAB R2025a + Deep Learning Toolbox
%
% Role de ce script :
%   1. Relire le split de P1, sans dependre de ses chemins Windows.
%   2. Evaluer chaque modele deja entraine (.mat), sur les MEMES images.
%   3. Calculer accuracy, precision, recall, macro-F1, F1 pondere,
%      balanced accuracy, nombre de parametres et Frugality Score.
%   4. Exporter matrices de confusion, erreurs et courbes si disponibles.
%
% IMPORTANT : data/test n'est JAMAIS utilise dans ce script.
% Ce script ne re-entraine aucun reseau.
%
% Lancer depuis le dossier du projet : evaluate_models

clearvars; clc;

%% A. PARAMETRES FACILES A MODIFIER
miniBatchSize = 32;              % Baisser a 8/16 si memoire limitee
executionEnvironment = 'auto';   % 'cpu', 'gpu' ou 'auto'
w = 0.3;                        % Valeur de la consigne du challenge
expectedValidationCount = 2659;

classNames = ["bread", "dairy", "dessert", "egg", "fried", "meat", ...
    "noodles-pasta", "rice", "seafood", "soup", "vegetable-fruit"];

% Fichiers cherches dans models/ (le premier trouve est utilise).
modelNames = ["LeNet-5", "MobileNetV2", "ResNet"];
modelCandidates = { ...
    {'lenet5_model.mat'}, ...
    {'mobilenetv2_model.mat'}, ...
    {'resnet18_model.mat'} ...
    };

%% B. CHEMINS RELATIFS A LA RACINE DU PROJET
projectRoot = locateProjectRoot(fileparts(mfilename('fullpath')));
modelDir = fullfile(projectRoot,'models');
resultDir = fullfile(projectRoot,'results','evaluation');
if ~isfolder(resultDir), mkdir(resultDir); end

csvVal = firstExistingFile({fullfile(projectRoot,'data','splits','validation_files.csv')});
trainDir = firstExistingFolder({fullfile(projectRoot,'data','train')});

if isempty(csvVal)
    error('CSV absent : data/splits/validation_files.csv');
end
if isempty(trainDir)
    error('Images absentes : remplir data/train/ avec les 11 dossiers de classes.');
end

%% C. VERIFIER LE SPLIT ET RECONSTRUIRE LES CHEMINS
% On ne conserve que (categorie, nom de fichier) du CSV, puis on
% reconstruit data/train/classe/image.jpg localement.
Tval = readtable(csvVal,'TextType','string');
if ~all(ismember({'File','Label'},Tval.Properties.VariableNames))
    error('Le CSV validation doit avoir les colonnes File et Label.');
end
fileOriginal = replace(string(Tval.File),char(92),'/');
yTrue = lower(strtrim(string(Tval.Label)));
N = numel(yTrue);
if N ~= expectedValidationCount
    error('Validation : %d lignes, attendu %d.',N,expectedValidationCount);
end
if any(~ismember(yTrue,classNames))
    error('Le CSV contient des categories inconnues.');
end

pathsVal = strings(N,1);
idsVal = strings(N,1);
for i=1:N
    [~,stem,ext] = fileparts(fileOriginal(i));
    idsVal(i) = yTrue(i) + "/" + string(stem) + string(ext);
    pathsVal(i) = string(fullfile(trainDir,char(yTrue(i)),char(string(stem)+string(ext))));
end
if numel(unique(idsVal))~=N
    error('Plusieurs lignes designent la meme image de validation.');
end

% Verification explicite : pas de fuite de donnees entre train et val.
trainCsv = fullfile(fileparts(csvVal),'train_files.csv');
if isfile(trainCsv)
    Ttrain = readtable(trainCsv,'TextType','string');
    if ~all(ismember({'File','Label'},Ttrain.Properties.VariableNames))
        error('Le CSV train doit avoir les colonnes File et Label.');
    end
    originalTrain = replace(string(Ttrain.File),char(92),'/');
    idsTrain = strings(height(Ttrain),1);
    for i=1:height(Ttrain)
        [~,stem,ext] = fileparts(originalTrain(i));
        idsTrain(i) = lower(strtrim(string(Ttrain.Label(i)))) + "/" + ...
            string(stem) + string(ext);
    end
    if numel(unique(idsTrain))~=numel(idsTrain)
        error('Doublons dans train_files.csv.');
    end
    if ~isempty(intersect(idsTrain,idsVal))
        error('FUITE DE DONNEES : une image est presente dans train ET validation.');
    end
    if numel(idsTrain)+N~=13296
        error('Train + validation ne totalisent pas 13296 images.');
    end
end

missing = ~isfile(pathsVal);
if any(missing)
    firstMissing = find(missing,1);
    error(['%d fichiers validation introuvables. Exemple : %s\n' ...
        'Copier le dataset officiel dans data/train/<classe>/*.jpg.'], ...
        nnz(missing),char(pathsVal(firstMissing)));
end

% PAS DE LABELS dans ce datastore : la prediction garde strictement
% l'ordre des lignes du CSV ; les vraies classes restent dans yTrue.
imdsVal = imageDatastore(cellstr(pathsVal));
fprintf('Validation : %d images / %d classes ; train-val disjoints.\n', ...
    N,numel(unique(yTrue)));

%% D. EVALUATION DES MODELES DISPONIBLES
results = struct('Model',{},'Accuracy',{},'MacroPrecision',{}, ...
    'MacroRecall',{},'MacroF1',{},'WeightedF1',{}, ...
    'BalancedAccuracy',{},'Parameters',{},'ParametersM',{}, ...
    'FrugalityScore',{},'InferenceSeconds',{},'ValidationImages',{}, ...
    'InputHeight',{},'InputWidth',{},'ModelFile',{});

for m=1:numel(modelNames)
    modelName = modelNames(m);
    candidatePaths = cellfun(@(nm) fullfile(modelDir,nm), ...
        modelCandidates{m},'UniformOutput',false);
    modelPath = firstExistingFile(candidatePaths);
    if isempty(modelPath)
        fprintf('[EN ATTENTE] %s : .mat pas encore dans models/.\n',modelName);
        continue;
    end

    fprintf('\n--- %s ---\n',modelName);
    try
        [net, classOrder, inputSize, saved] = loadClassificationModel(modelPath);
        fprintf('Modele : %s | entree : %d x %d\n', ...
            modelPath,inputSize(1),inputSize(2));

        % Pas de rotation / zoom / flip en validation : resize uniquement.
        ds = augmentedImageDatastore(inputSize(1:2),imdsVal, ...
            'ColorPreprocessing','gray2rgb');
        tStart = tic;
        yPred = predictLabels(net,ds,classOrder,miniBatchSize,executionEnvironment);
        seconds = toc(tStart);
        yPred = lower(strtrim(string(yPred(:))));

        if numel(yPred)~=N
            error('Predictions incompletes : %d au lieu de %d.',numel(yPred),N);
        end
        if any(~ismember(yPred,classNames))
            error(['Une sortie ne correspond pas aux 11 noms Food-11. ' ...
                'Verifier l''ordre des classes du reseau.']);
        end

        [~,iTrue] = ismember(yTrue,classNames);
        [~,iPred] = ismember(yPred,classNames);
        C = accumarray([iTrue(:),iPred(:)],1,[numel(classNames),numel(classNames)]);
        TP = diag(C);
        support = sum(C,2);
        predCounts = sum(C,1)';
        precision = safeDivide(TP,predCounts);
        recall = safeDivide(TP,support);
        f1 = safeDivide(2*precision.*recall,precision+recall);
        accuracy = sum(TP)/sum(C(:));
        macroPrecision = mean(precision);
        macroRecall = mean(recall);
        macroF1 = mean(f1);                % 11 classes a poids egal
        weightedF1 = sum(f1.*support)/sum(support);
        balancedAcc = mean(recall);        % identique au macro-recall

        numberParams = countModelParameters(net);
        if numberParams<=0
            error('Impossible de compter les parametres du reseau.');
        end
        R = numberParams/1e6;             % IMPORTANT : millions !
        frugality = macroF1 - w/(1+1/R);
        [~,modelBase,modelExt] = fileparts(modelPath);
        modelFilename = string(modelBase)+string(modelExt);
        results(end+1) = struct( ... %#ok<SAGROW>
            'Model',modelName,'Accuracy',accuracy, ...
            'MacroPrecision',macroPrecision,'MacroRecall',macroRecall, ...
            'MacroF1',macroF1,'WeightedF1',weightedF1, ...
            'BalancedAccuracy',balancedAcc, ...
            'Parameters',numberParams,'ParametersM',R, ...
            'FrugalityScore',frugality,'InferenceSeconds',seconds, ...
            'ValidationImages',N,'InputHeight',inputSize(1), ...
            'InputWidth',inputSize(2),'ModelFile',modelFilename);

        tag = char(regexprep(lower(modelName),'[^a-z0-9]','_'));
        perClass = table(classNames(:),support,precision,recall,f1, ...
            'VariableNames',{'Class','Support','Precision','Recall','F1'});
        writetable(perClass,fullfile(resultDir,['per_class_' tag '.csv']));
        predictionTable = table(pathsVal,yTrue,yPred, ...
            'VariableNames',{'File','TrueLabel','PredictedLabel'});
        writetable(predictionTable,fullfile(resultDir, ...
            ['validation_predictions_' tag '.csv']));
        writematrix(C,fullfile(resultDir,['confusion_counts_' tag '.csv']));
        save(fullfile(resultDir,['details_' tag '.mat']), ...
            'C','classNames','yTrue','yPred','precision','recall','f1','support');

        % Matrice normalisee par VRAIE classe : chaque ligne vaut 100 %.
        try
            fig = figure('Visible','off','Position',[50 50 1250 890]);
            cm = confusionchart(C,cellstr(classNames), ...
                'Normalization','row-normalized');
            cm.Title = char(modelName+" - validation (lignes = classes reelles)");
            exportgraphics(fig,fullfile(resultDir, ...
                ['confusion_' tag '.png']),'Resolution',160);
            close(fig);
        catch plotErr
            warning('Figure de confusion de %s : %s',modelName,plotErr.message);
            if exist('fig','var') && isgraphics(fig),close(fig);end
        end

        % Confusions les plus frequentes : utile pour l'analyse du rapport.
        Cerrors=C;
        Cerrors(1:size(Cerrors,1)+1:end)=0;
        [sortedErrors,linearIdx] = sort(Cerrors(:),'descend');
        keep = find(sortedErrors>0,5,'first');
        [rows,cols] = ind2sub(size(Cerrors),linearIdx(keep));
        errorPairs = table(classNames(rows),classNames(cols), ...
            sortedErrors(keep), ...
            safeDivide(sortedErrors(keep),support(rows)), ...
            'VariableNames',{'TrueClass','PredictedClass','Count','FractionOfTrueClass'});
        writetable(errorPairs,fullfile(resultDir,['top_errors_' tag '.csv']));

        % Si le .mat contient trainInfo, exporter les courbes d'apprentissage.
        try
            exportTrainingHistory(saved,modelName,resultDir,tag);
        catch histErr
            warning('Courbe apprentissage %s non disponible : %s', ...
                modelName,histErr.message);
        end

        fprintf(['Accuracy %.2f %% | macro-F1 %.4f | F1 pondere %.4f | ' ...
            '%.3f M params | Frugality %.4f | %.1f s\n'], ...
            100*accuracy,macroF1,weightedF1,R,frugality,seconds);
    catch ME
        warning('Evaluation de %s non effectuee : %s',modelName,ME.message);
        fprintf('Le prochain reseau sera teste si present.\n');
    end
end

if isempty(results)
    fprintf('\nAUCUN MODELE EVALUE : aucun .mat exploitable dans models/.\n');
    return;
end

T = struct2table(results);
T = sortrows(T,'FrugalityScore','descend');
writetable(T,fullfile(resultDir,'model_metrics.csv'));
fprintf('\n=== COMPARAISON PROVISOIRE/COMPLETE SUR VALIDATION ===\n');
disp(T(:,{'Model','Accuracy','MacroF1','WeightedF1', ...
    'BalancedAccuracy','ParametersM','FrugalityScore'}));
fprintf('Fichier : %s\n',fullfile(resultDir,'model_metrics.csv'));
fprintf('Attention : F1 et Frugality sont LOCAUX sur validation.\n');
fprintf('Le serveur du challenge mesure la performance sur le test officiel.\n');

%% FONCTIONS LOCALES - autonomes : pas de fichier supplementaire requis
function root = locateProjectRoot(here)
    roots = {here,fileparts(here),fileparts(fileparts(here))};
    for q=1:numel(roots)
        if isfolder(fullfile(roots{q},'models')) && ...
                isfolder(fullfile(roots{q},'data'))
            root=roots{q};return;
        end
    end
    error('Racine du projet Food-11 introuvable (dossiers models/ et data/).');
end
function p=firstExistingFile(candidates)
    p='';
    for q=1:numel(candidates)
        if isfile(candidates{q}),p=candidates{q};return;end
    end
end
function p=firstExistingFolder(candidates)
    p='';
    for q=1:numel(candidates)
        if isfolder(candidates{q}),p=candidates{q};return;end
    end
end
function out=safeDivide(num,den)
    out=zeros(size(num));
    good=den~=0;
    out(good)=num(good)./den(good);
end
function [net,order,inputSize,S]=loadClassificationModel(matPath)
    S=load(matPath);
    net=[];
    preferred={'trainedNet','net','trainedNetwork','network'};
    fields=[preferred,fieldnames(S)'];
    for q=1:numel(fields)
        fn=fields{q};
        if ~isfield(S,fn),continue;end
        candidate=S.(fn);
        if isa(candidate,'SeriesNetwork') || isa(candidate,'DAGNetwork') || ...
                isa(candidate,'dlnetwork')
            net=candidate;break;
        end
    end
    if isempty(net)
        error('Le fichier .mat ne contient pas de reseau entraine reconnu.');
    end
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
    for q=1:numel(net.Layers)
        if isprop(net.Layers(q),'InputSize')
            value=net.Layers(q).InputSize;
            if isnumeric(value) && numel(value)>=2
                inputSize=double(value);break;
            end
        end
    end
    if isempty(inputSize) && isfield(S,'inputSize')
        inputSize=double(S.inputSize);
    end
    if numel(inputSize)<2
        error('Dimensions de l''entree inconnues : sauver inputSize dans .mat.');
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
            error('Dimensions des scores incompatibles avec classNames.');
        end
        y=classOrder(idx(:));
    else
        y=classify(net,ds,'MiniBatchSize',batchSize, ...
            'ExecutionEnvironment',environment);
    end
end
function n=countModelParameters(net)
    n=0;
    if isa(net,'dlnetwork')
        learnables=net.Learnables;
        for q=1:height(learnables)
            n=n+numel(learnables.Value{q});
        end
    else
        props={'Weights','Bias','Scale','Offset','InputWeights', ...
            'RecurrentWeights'};
        for q=1:numel(net.Layers)
            layer=net.Layers(q);
            for j=1:numel(props)
                if isprop(layer,props{j})
                    val=layer.(props{j});
                    if isnumeric(val) || isa(val,'dlarray')
                        n=n+numel(val);
                    end
                end
            end
        end
    end
end
function exportTrainingHistory(S,modelName,resultDir,tag)
    if ~isfield(S,'trainInfo'),return;end
    info=S.trainInfo;
    if isobject(info) && isprop(info,'TrainingHistory')
        % Objet TrainingInfo renvoye par trainnet : deux tables separees
        th=info.TrainingHistory; vh=info.ValidationHistory;
        if ~ismember('Accuracy',th.Properties.VariableNames),return;end
        x=double(th.Iteration); trainAcc=double(th.Accuracy);
        if ~isempty(vh) && ismember('Accuracy',vh.Properties.VariableNames)
            xv=double(vh.Iteration); valAcc=double(vh.Accuracy);
        else
            xv=[]; valAcc=[];
        end
    else
        % Ancien format (struct / table de trainNetwork)
        if ~isstruct(info) && ~istable(info),return;end
        if istable(info), info=table2struct(info,'ToScalar',true); end
        if ~isfield(info,'TrainingAccuracy') || ~isfield(info,'ValidationAccuracy')
            return;
        end
        trainAcc=double(info.TrainingAccuracy(:));
        valAcc=double(info.ValidationAccuracy(:));
        if isfield(info,'Iteration')
            x=double(info.Iteration(:));
        else
            x=(1:numel(trainAcc))';
        end
        if numel(x)~=numel(trainAcc),return;end
        valid=isfinite(valAcc); xv=x(valid); valAcc=valAcc(valid);
    end
    fig=figure('Visible','off','Position',[50 50 1050 550]);
    plot(x,trainAcc,'-','LineWidth',1.6);hold on;
    if ~isempty(valAcc)
        plot(xv,valAcc,'o-','LineWidth',1.4);
    end
    xlabel('Iterations');ylabel('Accuracy (%)');
    title(char(modelName+" - entrainement / validation"));
    legend({'Train','Validation'},'Location','best');grid on;
    exportgraphics(fig,fullfile(resultDir,['learning_curve_' tag '.png']), ...
        'Resolution',160);
    close(fig);
end
