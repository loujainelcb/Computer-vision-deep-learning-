%% EVALUATION COMPLETE : figures, tableau et JSON en une commande
%   >> run_evaluation
%
% Prerequis : images dans data/train et data/test, modeles dans models/.
% Sorties :
%   results/evaluation/    metriques detaillees, matrices, courbes par modele
%   results/presentation/  tableau + figures prets pour les slides
%   results/evaluation/submission_food11.json   fichier a soumettre

setup_project
evaluate_models      % metriques sur la validation -> model_metrics.csv
compare_models       % classement + dashboard + heatmap F1
make_presentation    % tableau recapitulatif + figures des 3 modeles
generate_json        % predictions du meilleur modele sur le test
