function cfg = trainingConfig()
% TRAININGCONFIG Protocole d'entrainement COMMUN aux trois modeles.
%
% Pour que la comparaison soit juste, seule l'architecture change d'un
% modele a l'autre. Tout ce qui est ici s'applique a LeNet-5, MobileNetV2
% et ResNet-18. Modifier une valeur ici la modifie pour les trois.

% --- Reproductibilite ---
cfg.seed = 42;

% --- Budget d'entrainement (identique pour tous) ---
cfg.maxEpochs     = 15;
cfg.miniBatchSize = 32;
cfg.optimizer     = "adam";
cfg.outputNetwork = "best-validation";   % on garde le meilleur epoch en validation
cfg.executionEnvironment = "auto";       % GPU si disponible, sinon CPU

% --- Courbes d'entrainement (makeCourseProgressOutputFcn, memes pour les 3) ---
cfg.plotSmoothingWindow = 10;            % moyenne glissante sur 10 iterations

% --- Learning rate : seule difference autorisee, selon le point de depart ---
cfg.learnRate.scratch  = 1e-3;   % LeNet-5 : poids aleatoires
cfg.learnRate.transfer = 1e-4;   % MobileNetV2 / ResNet-18 : poids ImageNet

% --- Transfer learning (meme strategie pour MobileNetV2 et ResNet-18) ---
% true  : seule la nouvelle couche de classification est entrainee (rapide)
% false : tout le reseau est affine avec le learning rate "transfer"
cfg.freezeBackbone = true;

% --- Desequilibre des classes (cf. docs/preprocessing.md) ---
% Poids inverses a la frequence de chaque classe dans le train :
% rice / dairy comptent autant que dessert dans la loss -> meilleur macro-F1.
cfg.useClassWeights = true;

% --- Data augmentation definie par P1 (train uniquement) ---
cfg.aug.rotation    = [-15 15];    % degres
cfg.aug.reflectionX = true;        % flip horizontal
cfg.aug.scale       = [0.9 1.2];   % zoom
cfg.aug.translation = [-10 10];    % pixels, en X et en Y
cfg.aug.brightness  = [0.8 1.2];   % facteur multiplicatif
end
