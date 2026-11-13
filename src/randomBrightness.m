function J = randomBrightness(I, range)
% RANDOMBRIGHTNESS Multiplie l'image par un facteur aleatoire dans range
% (par defaut [0.8 1.2], cf. trainingConfig). Resultat en uint8.
if nargin < 2
    cfg = trainingConfig();
    range = cfg.aug.brightness;
end
f = range(1) + (range(2) - range(1)) * rand;
J = uint8(double(I) * f);
end
