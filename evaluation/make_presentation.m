%% PRESENTATION - tableau recapitulatif et figures pour les slides
% A lancer APRES evaluate_models (et idealement compare_models).
% Lit results/evaluation/*.csv et models/*.mat, ecrit tout dans
% results/presentation/ :
%   tableau_recapitulatif.png / .csv / .md   tableau des 3 modeles
%   confusion_3_modeles.png                  3 matrices de confusion cote a cote
%   courbes_validation_3_modeles.png         accuracy / loss de validation par epoch
%   f1_par_classe.png                        F1 de chaque classe pour chaque modele
%   + copie des figures utiles deja produites (dataset, augmentation,
%     progression d'entrainement, dashboard, heatmap)

clearvars; clc; close all;

root    = fileparts(fileparts(mfilename('fullpath')));
evalDir = fullfile(root, 'results', 'evaluation');
outDir  = fullfile(root, 'results', 'presentation');
if ~isfolder(outDir), mkdir(outDir); end

metricPath = fullfile(evalDir, 'model_metrics.csv');
if ~isfile(metricPath)
    error('model_metrics.csv absent : lancer evaluate_models d''abord.');
end
T = readtable(metricPath, 'TextType', 'string');
if isempty(T), error('Aucun modele evalue.'); end
T = sortrows(T, 'FrugalityScore', 'descend');
n = height(T);
tags = strings(n,1);
for k = 1:n
    tags(k) = string(regexprep(lower(char(T.Model(k))), '[^a-z0-9]', '_'));
end
classes = ["bread","dairy","dessert","egg","fried","meat", ...
    "noodles-pasta","rice","seafood","soup","vegetable-fruit"];
colors = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19; 0.49 0.18 0.56];

%% 1. TABLEAU RECAPITULATIF
msPerImage = 1000 * T.InferenceSeconds ./ T.ValidationImages;
P = table(T.Model, round(T.ParametersM, 2), round(100*T.Accuracy, 1), ...
    round(T.MacroF1, 3), round(100*T.BalancedAccuracy, 1), ...
    round(T.FrugalityScore, 3), round(msPerImage, 1), ...
    'VariableNames', {'Modele','Parametres_M','Accuracy_pct','MacroF1', ...
    'BalancedAccuracy_pct','FrugalityScore','ms_par_image'});
writetable(P, fullfile(outDir, 'tableau_recapitulatif.csv'));

headers = ["Modèle","Paramètres (M)","Accuracy","Macro-F1", ...
    "Balanced acc.","Frugality","ms / image"];
cells = [P.Modele, compose("%.2f", P.Parametres_M), ...
    compose("%.1f %%", P.Accuracy_pct), compose("%.3f", P.MacroF1), ...
    compose("%.1f %%", P.BalancedAccuracy_pct), ...
    compose("%.3f", P.FrugalityScore), compose("%.1f", P.ms_par_image)];

% Markdown (a coller dans un rapport / un README)
fid = fopen(fullfile(outDir, 'tableau_recapitulatif.md'), 'w', 'n', 'UTF-8');
fprintf(fid, '| %s |\n', strjoin(headers, ' | '));
fprintf(fid, '|---|%s\n', repmat('---:|', 1, numel(headers)-1));
for i = 1:n
    row = cells(i,:);
    if i == 1, row = "**" + row + "**"; end   % meilleur modele en gras
    fprintf(fid, '| %s |\n', strjoin(row, ' | '));
end
fclose(fid);

% Image (a poser directement sur une slide)
nc = numel(headers);
fig = figure('Visible','off','Color','w','Position',[50 50 1500 170+75*n]);
ax = axes(fig, 'Position', [0.01 0.02 0.98 0.96]);
axis(ax, 'off'); hold(ax, 'on'); set(ax, 'YDir', 'reverse');
xlim(ax, [0 nc]); ylim(ax, [0 n+1.7]);
rectangle(ax, 'Position', [0 0 nc 1], 'FaceColor', [0.13 0.27 0.45], 'EdgeColor', 'none');
rectangle(ax, 'Position', [0 1 nc 1], 'FaceColor', [0.86 0.95 0.86], 'EdgeColor', 'none');
for j = 1:nc
    text(ax, j-0.5, 0.5, headers(j), 'Color', 'w', 'FontWeight', 'bold', ...
        'FontSize', 15, 'HorizontalAlignment', 'center');
end
for i = 1:n
    weight = 'normal'; if i == 1, weight = 'bold'; end
    for j = 1:nc
        text(ax, j-0.5, i+0.5, cells(i,j), 'FontSize', 15, 'FontWeight', weight, ...
            'HorizontalAlignment', 'center');
    end
    plot(ax, [0 nc], [i+1 i+1], 'Color', [0.8 0.8 0.8]);
end
text(ax, 0.05, n+1.35, ['Mesures sur la validation (2 659 images). Classement par ' ...
    'Frugality Score = macro-F1 - 0.3 / (1 + 1/R), R en millions. Meilleur modèle en vert.'], ...
    'FontSize', 11, 'Color', [0.35 0.35 0.35]);
exportgraphics(fig, fullfile(outDir, 'tableau_recapitulatif.png'), 'Resolution', 170);
close(fig);
fprintf('Tableau : %s\n', fullfile(outDir, 'tableau_recapitulatif.png'));

%% 2. MATRICES DE CONFUSION COTE A COTE
fig = figure('Visible','off','Color','w','Position',[40 40 620*n 680]);
tl = tiledlayout(fig, 1, n, 'TileSpacing', 'compact', 'Padding', 'compact');
cmap = [linspace(1,0.08,256)' linspace(1,0.27,256)' linspace(1,0.55,256)'];
for k = 1:n
    ax = nexttile(tl);
    p = fullfile(evalDir, "confusion_counts_" + tags(k) + ".csv");
    if ~isfile(p)
        axis(ax, 'off'); title(ax, T.Model(k) + " (absent)"); continue;
    end
    C = readmatrix(p);
    Cn = 100 * C ./ max(sum(C, 2), 1);       % % par vraie classe
    imagesc(ax, Cn, [0 100]); colormap(ax, cmap); axis(ax, 'square');
    for i = 1:size(Cn,1)
        for j = 1:size(Cn,2)
            if Cn(i,j) >= 5
                c = 'k'; if Cn(i,j) > 55, c = 'w'; end
                text(ax, j, i, sprintf('%.0f', Cn(i,j)), 'Color', c, ...
                    'FontSize', 8, 'HorizontalAlignment', 'center');
            end
        end
    end
    xticks(ax, 1:11); xticklabels(ax, cellstr(classes)); xtickangle(ax, 45);
    yticks(ax, 1:11); yticklabels(ax, cellstr(classes));
    xlabel(ax, 'Classe prédite');
    if k == 1, ylabel(ax, 'Classe réelle'); end
    title(ax, sprintf('%s | macro-F1 %.3f', T.Model(k), T.MacroF1(k)));
end
cb = colorbar(ax); cb.Label.String = '% de la classe réelle';
title(tl, 'Matrices de confusion sur la validation (lignes normalisées à 100 %)');
exportgraphics(fig, fullfile(outDir, 'confusion_3_modeles.png'), 'Resolution', 170);
close(fig);

%% 3. COURBES DE VALIDATION DES 3 MODELES SUR LE MEME GRAPHE
fig = figure('Visible','off','Color','w','Position',[40 40 1500 560]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
axA = nexttile(tl); hold(axA, 'on'); grid(axA, 'on');
axL = nexttile(tl); hold(axL, 'on'); grid(axL, 'on');
nPlotted = 0;
for k = 1:n
    try
        mp = fullfile(root, 'models', char(T.ModelFile(k)));
        vars = whos('-file', mp);
        if ~ismember('trainInfo', {vars.name}), continue; end
        S = load(mp, 'trainInfo'); info = S.trainInfo;
        if ~(isobject(info) && isprop(info, 'ValidationHistory')), continue; end
        vh = info.ValidationHistory; th = info.TrainingHistory;
        if isempty(vh), continue; end
        if ismember('Epoch', vh.Properties.VariableNames)
            x = double(vh.Epoch);
        else
            x = double(vh.Iteration) / max(double(th.Iteration)) * max(double(th.Epoch));
        end
        col = colors(mod(k-1, size(colors,1)) + 1, :);
        if ismember('Accuracy', vh.Properties.VariableNames)
            plot(axA, x, double(vh.Accuracy), 'o-', 'Color', col, ...
                'LineWidth', 1.8, 'DisplayName', T.Model(k));
        end
        if ismember('Loss', vh.Properties.VariableNames)
            plot(axL, x, double(vh.Loss), 'o-', 'Color', col, ...
                'LineWidth', 1.8, 'DisplayName', T.Model(k));
        end
        nPlotted = nPlotted + 1;
    catch err
        warning('Courbe de %s ignoree : %s', T.Model(k), err.message);
    end
end
if nPlotted > 0
    xlabel(axA, 'Epoch'); ylabel(axA, 'Accuracy de validation (%)');
    title(axA, 'Accuracy de validation'); legend(axA, 'Location', 'southeast');
    xlabel(axL, 'Epoch'); ylabel(axL, 'Loss de validation');
    title(axL, 'Loss de validation'); legend(axL, 'Location', 'northeast');
    title(tl, 'Apprentissage des modèles (même protocole, même validation)');
    exportgraphics(fig, fullfile(outDir, 'courbes_validation_3_modeles.png'), 'Resolution', 170);
else
    fprintf(['Courbes de validation non produites : aucun .mat ne contient ' ...
        'trainInfo (modeles a reentrainer avec training/).\n']);
end
close(fig);

%% 4. F1 PAR CLASSE (barres groupees)
F = nan(numel(classes), n);
for k = 1:n
    p = fullfile(evalDir, "per_class_" + tags(k) + ".csv");
    if ~isfile(p), continue; end
    Pc = readtable(p, 'TextType', 'string');
    [ok, idx] = ismember(classes, string(Pc.Class));
    F(ok, k) = Pc.F1(idx(ok));
end
fig = figure('Visible','off','Color','w','Position',[40 40 1500 600]);
ax = axes(fig);
b = bar(ax, categorical(cellstr(classes), cellstr(classes)), F, 'grouped');
for k = 1:n, b(k).FaceColor = colors(mod(k-1, size(colors,1)) + 1, :); end
ylim(ax, [0 1]); ylabel(ax, 'F1-score'); grid(ax, 'on');
legend(ax, cellstr(T.Model), 'Location', 'southoutside', 'Orientation', 'horizontal');
title(ax, 'F1 par catégorie sur la validation (classes rares : rice, dairy, noodles-pasta)');
exportgraphics(fig, fullfile(outDir, 'f1_par_classe.png'), 'Resolution', 170);
close(fig);

%% 5. RASSEMBLER LES AUTRES FIGURES UTILES DANS LE MEME DOSSIER
toCopy = { ...
    fullfile(root,'results','preprocessing','class_distribution.png'),        '01_distribution_classes.png'; ...
    fullfile(root,'results','preprocessing','sample_images.png'),             '02_exemples_images.png'; ...
    fullfile(root,'results','preprocessing','augmentation_examples.png'),     '03_augmentation.png'; ...
    fullfile(root,'results','training','lenet5_progress.png'),                '04_progression_lenet5.png'; ...
    fullfile(root,'results','training','mobilenetv2_progress.png'),           '05_progression_mobilenetv2.png'; ...
    fullfile(root,'results','training','finetuning','full','resnet18_full_ft_progress.png'), '06_progression_resnet18.png'; ...
    fullfile(evalDir,'comparison_dashboard.png'),                             '07_dashboard_comparaison.png'; ...
    fullfile(evalDir,'classwise_f1_heatmap.png'),                             '08_heatmap_f1.png'};
for i = 1:size(toCopy, 1)
    if isfile(toCopy{i,1})
        copyfile(toCopy{i,1}, fullfile(outDir, toCopy{i,2}));
    end
end

fprintf('\n=== FIGURES DE PRESENTATION ===\n');
listing = dir(fullfile(outDir, '*.png'));
fprintf('  %s\n', listing.name);
fprintf('Dossier : %s\n', outDir);
disp(P);
