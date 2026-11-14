
% ============ CODE POUR L'ARCHITECTURE LeNet-5 OPTIMISÉE ================

function layers = leNet5(inputSize, numClasses, InputLayer)
% leNet5 Construit l'architecture LeNet-5 étendue à 4 blocs
% inputSize  : Taille des images [Hauteur, Largeur, Canaux]
% numClasses : Nombre de classes à prédire
    if nargin < 1, inputSize = [128 128 3]; end
    if nargin < 2, numClasses = 11; end
    if nargin < 3
        InputLayer = imageInputLayer(inputSize, Name='input', ...
            Normalization='rescale-zero-one');
    end

    layers = [
        % -- COUCHE D'ENTRÉE --
        InputLayer

        % -- BLOC DE CONVOLUTION 1 (128x128 -> 64x64) --
        convolution2dLayer(5, 32, Padding='same', Name='conv1')
        batchNormalizationLayer(Name='batchnorm1')
        reluLayer(Name='relu1')
        maxPooling2dLayer(2, Stride=2, Name='pool1')

        % -- BLOC DE CONVOLUTION 2 (64x64 -> 32x32) --
        convolution2dLayer(5, 64, Padding='same', Name='conv2')
        batchNormalizationLayer(Name='batchnorm2')
        reluLayer(Name='relu2')
        maxPooling2dLayer(2, Stride=2, Name='pool2')

        % -- BLOC DE CONVOLUTION 3 (32x32 -> 16x16) --
        convolution2dLayer(3, 128, Padding='same', Name='conv3')
        batchNormalizationLayer(Name='batchnorm3')
        reluLayer(Name='relu3')
        maxPooling2dLayer(2, Stride=2, Name='pool3')

        % -- BLOC DE CONVOLUTION 4 (16x16 -> 8x8) --
        convolution2dLayer(3, 256, Padding='same', Name='conv4')
        batchNormalizationLayer(Name='batchnorm4')
        reluLayer(Name='relu4')
        averagePooling2dLayer(2, Stride=2, Name='pool4')

        % -- COUCHE PLEINEMENT CONNECTÉE (1 seule couche cachée) --
        fullyConnectedLayer(128, Name='fullyconnected1')
        batchNormalizationLayer(Name='batchnorm_fc1')
        reluLayer(Name='relu5')
        dropoutLayer(0.2, Name='drop1')

        % -- SORTIE --
        fullyConnectedLayer(numClasses, Name='fc_out')
        softmaxLayer(Name='softmax')
    ];
end
