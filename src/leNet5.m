% ============ CODE POUR L'architecture LeNet5================

function layers = leNet5(inputSize,numClasses,InputLayer)
 % leNet5 Construit l'architecture LeNet-5 
 % inputSize : Taille des images [Hauteur, Largeur, Canaux]
 % numClasses : Nombre de classes à prédire
    if nargin < 1, inputSize = [128 128 3]; end
    if nargin < 2, numClasses = 11; end
    if nargin < 3, InputLayer= imageInputLayer(inputSize, Name= 'input', ...
        Normalization = 'rescale-zero-one');end

    layers = [

        
        %--- COUCHE D'ENTREE---
        
        InputLayer
        
        % --BLOC DE CONOLUTION 1 
        % On a 6 filtres de taille 5x5. Padding = same (permet de conserver la taille de l'image
        convolution2dLayer(5, 6, Padding='same', Name='conv1')
        batchNormalizationLayer(Name='batchnorm1') %stabilise l'entrainement
        reluLayer(Name='relu1')  % Acitve les neurones , reLu est plus performante que tanh
        maxPooling2dLayer(2, Stride=2, Name='pool1') %sous_echantillonage par 2 en prenant le maximum
        
        % --BLOC DE CONOLUTION 2 
        % On a 16 filtres de taille 5x5. Padding = same (permet de conserver la taille de l'image
        convolution2dLayer(5, 16, Padding='same', Name='conv2')
        batchNormalizationLayer(Name='batchnorm2') %stabilise l'entrainement
        reluLayer(Name='relu2')  % Acitve les neurones , reLu est plus performante que tanh
        maxPooling2dLayer(2, Stride=2, Name='pool2') %sous_echantillonage par 2 en prenant le maximum
        
        % --BLOC DE CONOLUTION 3 
        % On a 120 filtres de taille 5x5. Ppour extraire les
        % caractéristiques plus complexes
        convolution2dLayer(5, 120, Padding='same', Name='conv3')
        batchNormalizationLayer(Name='batchnorm3') %stabilise l'entrainement
        reluLayer(Name='relu3')  % Acitve les neurones , reLu est plus performante que tanh
        averagePooling2dLayer(2, Stride=2, Name='pool3') %sous_echantillonage par 2 en prenant la moyenne
        
        %---Couches pleinement connectées (2)---
        fullyConnectedLayer(120, Name='fullyconnected1')
        reluLayer(Name='relu4')
        dropoutLayer(0.2, Name='drop1') %pour éviter le suraprentissage
        
        fullyConnectedLayer(84, Name='fullyconnected2')
        reluLayer(Name='relu5')
        dropoutLayer(0.2, Name='drop2')
        
        %----SORTIE-----------
        fullyConnectedLayer(numClasses, Name='fc_out')
        softmaxLayer(Name='softmax') %transforme les scores en proba

        ];
end