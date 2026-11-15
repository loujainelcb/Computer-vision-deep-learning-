%% MAKE_REPORT_FIGURES Figures compactes pour le rapport Food-11
% Utilise uniquement les resultats deja calcules dans results/evaluation/.
% A executer depuis MATLAB avec :
%   make_report_figures

clearvars;
close all;
clc;

projectRoot = fileparts(mfilename('fullpath'));
evaluationDir = fullfile(projectRoot, 'results', 'evaluation');
outputDir = fullfile(projectRoot, 'results', 'presentation', 'rapport');
if ~isfolder(outputDir)
    mkdir(outputDir);
end

sourceFiles = { ...
    'classwise_f1_comparison.csv', ...
    'confusion_counts_mobilenetv2.csv', ...
    'per_class_mobilenetv2.csv'};
for k = 1:numel(sourceFiles)
    sourcePath = fullfile(evaluationDir, sourceFiles{k});
    if ~isfile(sourcePath)
        error('Fichier de resultat absent : %s', sourcePath);
    end
end

f1Table = readtable(fullfile(evaluationDir, ...
    'classwise_f1_comparison.csv'), 'TextType', 'string');
perClassTable = readtable(fullfile(evaluationDir, ...
    'per_class_mobilenetv2.csv'), 'TextType', 'string');
confusionCounts = readmatrix(fullfile(evaluationDir, ...
    'confusion_counts_mobilenetv2.csv'));

reportModels = ["LeNet-5", "ResNet-18 gele", "ResNet-18 FT", "MobileNetV2"];
reportColumns = string(matlab.lang.makeValidName(cellstr(reportModels)));
requiredF1Columns = ["Class", reportColumns];
requiredPerClassColumns = ["Class", "Support", "Precision", "Recall", "F1"];
assertColumns(f1Table, requiredF1Columns, 'classwise_f1_comparison.csv');
assertColumns(perClassTable, requiredPerClassColumns, ...
    'per_class_mobilenetv2.csv');

classOrder = string(perClassTable.Class(:));
assert(numel(classOrder) == 11 && numel(unique(classOrder)) == 11, ...
    'per_class_mobilenetv2.csv doit contenir 11 classes distinctes.');
assert(size(f1Table, 1) == numel(classOrder), ...
    'classwise_f1_comparison.csv doit contenir 11 lignes.');
assert(isequal(string(f1Table.Class(:)), classOrder), ...
    ['Les classes de classwise_f1_comparison.csv et ' ...
     'per_class_mobilenetv2.csv ne sont pas dans le meme ordre.']);
assert(isequal(size(confusionCounts), [numel(classOrder), numel(classOrder)]), ...
    ['La matrice de confusion doit etre %d x %d, mais sa taille est %d x %d.'], ...
    numel(classOrder), numel(classOrder), size(confusionCounts, 1), ...
    size(confusionCounts, 2));
assert(all(isfinite(confusionCounts(:))) && ...
    all(confusionCounts(:) >= 0) && all(mod(confusionCounts(:), 1) == 0), ...
    'La matrice de confusion contient des valeurs invalides.');

shortClassNames = shortClassLabels(classOrder);

%% Figure 1 : heatmap compacte du F1-score par classe
f1Scores = zeros(height(f1Table), numel(reportColumns));
for k = 1:numel(reportColumns)
    f1Scores(:, k) = f1Table.(reportColumns(k));
end
assert(all(isfinite(f1Scores(:))) && all(f1Scores(:) >= 0) && ...
    all(f1Scores(:) <= 1), 'Les F1-scores doivent etre compris entre 0 et 1.');

fig = figure('Visible', 'off', 'Color', 'white', ...
    'Units', 'centimeters', 'Position', [2 2 13 9.5]);
ax = axes(fig);
imagesc(ax, f1Scores, [0 1]);
axis(ax, 'tight');
ax.FontName = 'Arial';
ax.FontSize = 8.5;
ax.Color = 'white';
ax.XColor = [0 0 0];
ax.YColor = [0 0 0];
ax.LineWidth = 0.7;
ax.Box = 'on';
ax.XTick = 1:numel(reportModels);
ax.YTick = 1:numel(classOrder);
ax.XTickLabel = {'LeNet-5', 'ResNet-18 gelé', 'ResNet-18 FT', 'MobileNetV2'};
ax.YTickLabel = cellstr(shortClassNames);
ax.TickLength = [0 0];
colormap(ax, parula(256));
cb = colorbar(ax);
cb.Ticks = 0:0.2:1;
cb.Label.String = 'F1-score';
cb.FontName = 'Arial';
cb.FontSize = 8;
cb.Color = [0 0 0];
for i = 1:size(f1Scores, 1)
    for j = 1:size(f1Scores, 2)
        textColor = contrastColor(f1Scores(i, j), parula(256));
        text(ax, j, i, sprintf('%.2f', f1Scores(i, j)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontName', 'Arial', 'FontSize', 8.5, ...
            'Color', textColor);
    end
end
set(ax, 'LooseInset', max(ax.TightInset, 0.01));
figurePath = fullfile(outputDir, 'f1_comparison_report.png');
exportgraphics(fig, figurePath, 'Resolution', 300);
close(fig);

%% Figure 2 : matrice de confusion normalisee de MobileNetV2
rowTotals = sum(confusionCounts, 2);
assert(all(rowTotals > 0), ...
    'Chaque classe reelle doit avoir au moins une image dans la matrice.');
confusionPercent = 100 * confusionCounts ./ rowTotals;

fig = figure('Visible', 'off', 'Color', 'white', ...
    'Units', 'centimeters', 'Position', [2 2 13 12.5]);
ax = axes(fig);
imagesc(ax, confusionPercent, [0 100]);
axis(ax, 'tight');
axis(ax, 'square');
ax.FontName = 'Arial';
ax.FontSize = 8;
ax.Color = 'white';
ax.XColor = [0 0 0];
ax.YColor = [0 0 0];
ax.LineWidth = 0.7;
ax.Box = 'on';
ax.XTick = 1:numel(classOrder);
ax.YTick = 1:numel(classOrder);
ax.XTickLabel = cellstr(shortClassNames);
ax.YTickLabel = cellstr(shortClassNames);
ax.TickLength = [0 0];
ax.XTickLabelRotation = 45;
colormap(ax, parula(256));
cb = colorbar(ax);
cb.Ticks = 0:20:100;
cb.Label.String = 'Pourcentage (%)';
cb.FontName = 'Arial';
cb.FontSize = 8;
cb.Color = [0 0 0];
for i = 1:size(confusionPercent, 1)
    for j = 1:size(confusionPercent, 2)
        if i == j || confusionPercent(i, j) >= 5
            textColor = contrastColor(confusionPercent(i, j) / 100, ...
                parula(256));
            text(ax, j, i, sprintf('%d %%', round(confusionPercent(i, j))), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'FontName', 'Arial', 'FontSize', 8, ...
                'Color', textColor);
        end
    end
end
xl = xlabel(ax, 'Prédit', 'FontSize', 9);
yl = ylabel(ax, 'Réel', 'FontSize', 9);
xl.Color = [0 0 0];
yl.Color = [0 0 0];
set(ax, 'LooseInset', max(ax.TightInset, 0.01));
figurePath = fullfile(outputDir, 'confusion_mobilenet_report.png');
exportgraphics(fig, figurePath, 'Resolution', 300);
close(fig);

% Controle explicite des deux valeurs demandees pour la matrice.
breadIndex = find(classOrder == "bread", 1);
eggIndex = find(classOrder == "egg", 1);
assert(~isempty(breadIndex) && ~isempty(eggIndex), ...
    'Les classes bread et egg sont necessaires au controle.');
assert(abs(confusionPercent(breadIndex, breadIndex) - 70.85) < 0.1, ...
    'La valeur Bread -> Bread ne correspond pas aux donnees source.');
assert(abs(confusionPercent(breadIndex, eggIndex) - 11.07) < 0.1, ...
    'La valeur Bread -> Egg ne correspond pas aux donnees source.');

fprintf('\nFigures du rapport generees dans : %s\n', outputDir);
fprintf(' - %s\n', fullfile(outputDir, 'f1_comparison_report.png'));
fprintf(' - %s\n', fullfile(outputDir, 'confusion_mobilenet_report.png'));

%% Fonctions locales
function assertColumns(T, required, fileName)
actual = string(T.Properties.VariableNames);
missing = required(~ismember(required, actual));
assert(isempty(missing), 'Colonnes manquantes dans %s : %s.', ...
    fileName, strjoin(missing, ', '));
end

function labels = shortClassLabels(classes)
classes = string(classes);
labels = strings(size(classes));
for k = 1:numel(classes)
    switch lower(classes(k))
        case "bread"
            labels(k) = "Bread";
        case "dairy"
            labels(k) = "Dairy";
        case "dessert"
            labels(k) = "Dessert";
        case "egg"
            labels(k) = "Egg";
        case "fried"
            labels(k) = "Fried";
        case "meat"
            labels(k) = "Meat";
        case "noodles-pasta"
            labels(k) = "Pasta";
        case "rice"
            labels(k) = "Rice";
        case "seafood"
            labels(k) = "Seafood";
        case "soup"
            labels(k) = "Soup";
        case "vegetable-fruit"
            labels(k) = "Veg./Fruit";
        otherwise
            error('Classe inconnue dans les resultats : %s', classes(k));
    end
end
end

function textColor = contrastColor(value, cmap)
index = min(size(cmap, 1), max(1, round(value * 255) + 1));
rgb = cmap(index, :);
luminance = 0.299 * rgb(1) + 0.587 * rgb(2) + 0.114 * rgb(3);
if luminance < 0.55
    textColor = [1 1 1];
else
    textColor = [0 0 0];
end
end
