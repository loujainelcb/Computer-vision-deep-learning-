function imds = load_split(csvPath, localTrainDir)
    % LOAD_SPLIT Charge un CSV de data/splits avec gestion robuste des délimiteurs CSV

    % 1. Lecture brute avec readcell (force le séparateur virgule)
    rawContent = readcell(csvPath, 'Delimiter', ',');

    % 2. Détection de l'en-tête ("File", "Label")
    firstCell = string(rawContent{1, 1});
    if strcmpi(firstCell, "File") || strcmpi(firstCell, "Label")
        startRow = 2;
    else
        startRow = 1;
    end

    numFiles = size(rawContent, 1) - startRow + 1;
    newPaths = cell(numFiles, 1);
    labels   = cell(numFiles, 1);

    % 3. Reconstruction propre de chaque chemin d'accès
    for k = 1:numFiles
        i = k + startRow - 1;
        
        origPathStr = string(rawContent{i, 1});
        labelStr    = string(rawContent{i, 2});

        % Extrait le nom du fichier image (ex: "933.jpg")
        [~, fname, fext] = fileparts(origPathStr);
        imageName = char(fname + fext);

        % Nettoie le nom de la catégorie (ex: "vegetable-fruit")
        categoryName = strtrim(char(labelStr));

        % Reconstruit le chemin valide localement : "data/train/vegetable-fruit/933.jpg"
        newPaths{k} = char(fullfile(localTrainDir, categoryName, imageName));
        labels{k}   = categoryName;
    end

    % 4. Diagnostic d'erreur précis en cas de fichier manquant
    if ~exist(newPaths{1}, 'file')
        fprintf('\n================ ERREUR DIAGNOSTIC ================\n');
        fprintf('Chemin d origin dans CSV : %s\n', string(rawContent{startRow, 1}));
        fprintf('Label lu dans CSV        : %s\n', string(rawContent{startRow, 2}));
        fprintf('Chemin local reconstruit : %s\n', newPaths{1});
        fprintf('Dossier cherche          : %s\n', localTrainDir);
        fprintf('====================================================\n\n');
        error('Fichier introuvable. Vérifie le diagnostic ci-dessus.');
    end

    % 5. Création de l'imageDatastore MATLAB
    imds = imageDatastore(newPaths, 'Labels', categorical(labels));
end