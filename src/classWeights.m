function [w, classNames] = classWeights(labels)
% CLASSWEIGHTS Poids par classe inverses a la frequence :
%   w_k = N / (K * n_k)   -> moyenne ponderee egale a 1
% L'ordre de w suit categories(labels), c'est-a-dire l'ordre des sorties
% du reseau quand on l'entraine avec trainnet sur ces labels.
classNames = categories(labels);
counts = countcats(labels);
w = numel(labels) ./ (numel(classNames) * counts);
w = w(:)';
end
