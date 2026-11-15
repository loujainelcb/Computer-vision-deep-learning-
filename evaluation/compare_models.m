%% P4 - COMPARAISON DES MODELES FOOD-11 (sur la validation)
% A executer apres evaluate_models.m.
% Cree un tableau pour le rapport et deux figures :
%   - comparison_dashboard.png : scores / frugality / params / compromis
%   - classwise_f1_heatmap.png : F1 par classe pour chaque reseau
%
% Les performances sont mesurees sur la VALIDATION et non le test officiel.

clearvars; clc;

%% A. CONFIGURATION
plannedModels = ["LeNet-5","MobileNetV2","ResNet-18 FT","ResNet-18 gele"];
projectRoot = locateProjectRoot(fileparts(mfilename('fullpath')));
resultDir = fullfile(projectRoot,'results','evaluation');
metricPath = fullfile(resultDir,'model_metrics.csv');
if ~isfile(metricPath)
    error('model_metrics.csv absent : lancer evaluate_models.m d''abord.');
end

%% B. CLASSEMENT EN FONCTION DU FRUGALITY SCORE
T=readtable(metricPath,'TextType','string');
required={'Model','Accuracy','MacroF1','WeightedF1','BalancedAccuracy', ...
    'ParametersM','FrugalityScore','ModelFile'};
if ~all(ismember(required,T.Properties.VariableNames))
    error('Colonnes manquantes dans model_metrics.csv.');
end
if isempty(T),error('Aucun reseau evalue.');end
T=sortrows(T,'FrugalityScore','descend');
T.Rank=(1:height(T))';
T=movevars(T,'Rank','Before','Model');
writetable(T,fullfile(resultDir,'model_ranking.csv'));

% Tant qu'il manque des modeles, le classement est PROVISOIRE.
allModelsReady=all(ismember(plannedModels,string(T.Model)));
if ~allModelsReady
    missing=plannedModels(~ismember(plannedModels,string(T.Model)));
    warning('Comparaison PROVISOIRE. Modeles pas encore evalues : %s', ...
        strjoin(missing,', '));
end
bestModelName=string(T.Model(1));
bestModelFile=string(T.ModelFile(1));
bestValidationF1=T.MacroF1(1);
bestValidationFrugality=T.FrugalityScore(1);
save(fullfile(resultDir,'best_model_choice.mat'), ...
    'bestModelName','bestModelFile','bestValidationF1', ...
    'bestValidationFrugality','allModelsReady');

disp('CLASSEMENT (validation) :');
disp(T(:,{'Rank','Model','Accuracy','MacroF1','WeightedF1', ...
    'BalancedAccuracy','ParametersM','FrugalityScore'}));
fprintf('Modele actuellement en tete : %s | Frugality = %.4f\n', ...
    bestModelName,bestValidationFrugality);

%% C. FIGURE 1 : TABLEAU DE BORD EN UNE SEULE FIGURE
% Separer score et parametres dans des axes distincts est essentiel :
% les echelles ne sont pas les memes (F1 ~ 0-1 ; parametres en millions).
try
    fig=figure('Visible','off','Position',[40 40 1450 970]);
    layout=tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');
    shortNames=categorical(cellstr(T.Model));
    shortNames=reordercats(shortNames,cellstr(T.Model));

    nexttile(layout,1);
    bar(shortNames,[T.Accuracy,T.MacroF1,T.BalancedAccuracy]);
    ylabel('Score (0 a 1)');ylim([0 1]);grid on;
    title('Qualite de classification (validation)');
    legend({'Accuracy','Macro-F1','Balanced accuracy'},'Location','best');

    nexttile(layout,2);
    bar(shortNames,T.FrugalityScore);
    ylabel('Frugality Score');grid on;
    title('Critere du challenge (plus haut = mieux)');

    nexttile(layout,3);
    bar(shortNames,T.ParametersM);
    ylabel('Millions de parametres');grid on;
    title('Taille du reseau (plus petit = plus frugal)');

    nexttile(layout,4);
    scatter(T.ParametersM,T.MacroF1,120,'filled');hold on;
    for j=1:height(T)
        text(T.ParametersM(j),T.MacroF1(j),"  "+string(T.Model(j)), ...
            'FontSize',10,'Interpreter','none');
    end
    xlabel('Parametres (millions)');ylabel('Macro-F1');ylim([0 1]);
    grid on;title('Compromis precision / complexite');

    sgtitle(layout,'Food-11 | Comparaison des modeles sur la validation');
    exportgraphics(fig,fullfile(resultDir,'comparison_dashboard.png'), ...
        'Resolution',170);
    close(fig);
catch plotErr
    warning('Impossible de creer le dashboard : %s',plotErr.message);
    if exist('fig','var') && isgraphics(fig),close(fig);end
end

%% D. FIGURE 2 : F1 POUR CHACUNE DES 11 CATEGORIES
% Important car le dataset n'est pas equilibre : le riz est peu represente.
try
    classes=["bread","dairy","dessert","egg","fried","meat", ...
        "noodles-pasta","rice","seafood","soup","vegetable-fruit"];
    classScores=nan(numel(classes),height(T));
    for j=1:height(T)
        tag=char(regexprep(lower(string(T.Model(j))),'[^a-z0-9]','_'));
        classPath=fullfile(resultDir,['per_class_' tag '.csv']);
        if ~isfile(classPath)
            warning('Detail par classe manquant : %s',classPath);
            continue;
        end
        P=readtable(classPath,'TextType','string');
        for i=1:numel(classes)
            k=find(string(P.Class)==classes(i),1);
            if ~isempty(k),classScores(i,j)=P.F1(k);end
        end
    end

    fig=figure('Visible','off','Position',[60 60 1050 760]);
    imagesc(classScores,[0 1]);colormap(parula);colorbar;
    xticks(1:height(T));xticklabels(cellstr(T.Model));
    yticks(1:numel(classes));yticklabels(cellstr(classes));
    ylabel('Categories Food-11');xlabel('Modele');
    title('F1-score par categorie (validation)');
    for i=1:numel(classes)
        for j=1:height(T)
            if ~isnan(classScores(i,j))
                text(j,i,sprintf('%.2f',classScores(i,j)), ...
                    'HorizontalAlignment','center','FontWeight','bold');
            end
        end
    end
    exportgraphics(fig,fullfile(resultDir,'classwise_f1_heatmap.png'), ...
        'Resolution',170);
    close(fig);
    heatTable=array2table(classScores,'VariableNames', ...
        matlab.lang.makeValidName(cellstr(T.Model)));
    heatTable=addvars(heatTable,classes(:),'Before',1,'NewVariableNames','Class');
    writetable(heatTable,fullfile(resultDir,'classwise_f1_comparison.csv'));
catch plotErr
    warning('Impossible de creer heatmap F1 : %s',plotErr.message);
    if exist('fig','var') && isgraphics(fig),close(fig);end
end

fprintf('\nFichiers du rapport dans %s\n',resultDir);
fprintf('Note : le score du site peut differer (autres images de test).\n');
fprintf('La formule locale utilise macro-F1 comme estimation du F1 moyen.\n');

%% FONCTION LOCALE
function root=locateProjectRoot(here)
    dirs={here,fileparts(here),fileparts(fileparts(here))};
    for j=1:numel(dirs)
        if isfolder(fullfile(dirs{j},'models')) && ...
                isfolder(fullfile(dirs{j},'data'))
            root=dirs{j};return;
        end
    end
    error('Racine du projet introuvable.');
end
