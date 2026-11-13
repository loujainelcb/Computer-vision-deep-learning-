function net = freezeBackbone(net)
% FREEZEBACKBONE Gele tous les poids pre-entraines (learn rate factor = 0)
% sauf ceux de la derniere couche apprenable, c'est-a-dire la nouvelle
% couche de classification a 11 classes ajoutee par imagePretrainedNetwork.
learnables = net.Learnables;
lastLayer = learnables.Layer(end);
for i = 1:height(learnables)
    factor = double(learnables.Layer(i) == lastLayer);
    net = setLearnRateFactor(net, learnables.Layer(i), learnables.Parameter(i), factor);
end
fprintf('Backbone gele : seule la couche "%s" est entrainee.\n', lastLayer);
end
