function [shapeRegressor] = fastFacealign_train(images, groundTruth, varargin)
    addpath(genpath('./libs'));
    
    %% --------------------------------------------------------------------
    %  -- LOAD ALGORITHM PARAMETERS
    %  --------------------------------------------------------------------
    
    params = {int32(5), int32(200), int32(5), single(0.06), int32(8), int32(20), int32(32), false};
    [numStages, numTrees, treeDepth, LAMBDA, NUM_FOLDS, NUM_AUGMENTS, NUM_SAMPLE_PIXELS, estimatesSet] = deal(params{:});
    
    for i = 1:2:length(varargin)
        k = varargin{i};
        v = varargin{i+1};
        if strcmp(k, 'numStages')
            numStages      = int32(v);
        elseif strcmp(k, 'numTrees')
            numTrees       = int32(v);
        elseif strcmp(k, 'treeDepth')
            treeDepth      = int32(v);
        elseif strcmp(k, 'lambda')
            LAMBDA         = single(v);
        elseif strcmp(k, 'numFolds')
            NUM_FOLDS      = int32(v);
        elseif strcmp(k, 'numAugments')
            NUM_AUGMENTS   = int32(v);
        elseif strcmp(k, 'model')
            shapeRegressor = v;
        elseif strcmp(k, 'lmkEstimates')
            lmkEstimates = v;
            estimatesSet = true;
        end
    end
    
    [imH,imW,~]             = size(images);
    [numLandmarks,~,numImg] = size(groundTruth);
    numLmkTrees             = ceil(numTrees / numLandmarks);
    numFeatsTree            = 2^treeDepth;
    numFeatsTotal           = numLandmarks * numLmkTrees * numFeatsTree;
    
    %% --------------------------------------------------------------------
    %  -- STAGES TRAINING METHODS
    %  --------------------------------------------------------------------
    
    function [stage, lmkUpdates] = generateStage(groundTruth, lmkEstimates)
        stage = struct( ...
            'trees', [], ...
            'weights', [] ...
        );
        
        % Find landmark displacements
        lmkDisplacements = groundTruth - lmkEstimates;
        
        % Find mean shape alignment transforms, prepare regression targets
        transforms = zeros(2, 2*nAug, 'single');
        targets    = zeros(nAug, 2*numLandmarks, 'single');
        for iE = 1:nAug
            tfm = pinvMeanShape * lmkEstimates(:,:,iE);
            transforms(:,int32([1 2])+2*(iE-1)) = tfm(1:2,:);
            lmkDisplacements(:,:,iE) = lmkDisplacements(:,:,iE) * pinv(tfm(1:2,:));
            targets(iE,:) = reshape(lmkDisplacements(:,:,iE)', 1, []);
        end
        
        % Generate candidate radii for the landmark regions
        nerrDists = sqrt(sum(lmkDisplacements .^2, 2));
        testRadii = 1.25 * std(nerrDists(:)) * (0.8:0.1:1.2);
        
        % Perform 10-fold cross-validation to select optimal region radius
        errors = zeros(numel(testRadii), 1, 'single');
        for iR = 1:numel(testRadii)
            [~, features] = generateRegressionTrees(transforms, lmkDisplacements, testRadii(iR));
            [~, pred]  = linregDualcoord(features, targets, LAMBDA, NUM_FOLDS);
            %errors(iR) = mean(sqrt(sum((targets - pred) .^ 2, 2)));
            errors(iR) = mean(sum((targets - pred) .^ 2, 2));
        end
        
        % Train final regressor
        [~,iR] = min(errors);
        [stage.trees, features] = generateRegressionTrees(transforms, lmkDisplacements, testRadii(iR));
        [stage.weights, lmkUpdates] = linregDualcoord(features, targets, LAMBDA);
        disp(['Training L2 MSE :: ' num2str(mean(sum((targets - lmkUpdates) .^ 2, 2)))]);
        
        % Bring landmark updates to transformed space
        for iE = 1:nAug
            lmkUpdates(iE,:) = reshape((reshape(lmkUpdates(iE,:), 2, [])' * transforms(:,int32([1 2])+2*(iE-1)))', 1, []);
        end
    end

    function [trees, features] = generateRegressionTrees(transforms, lmkDisplacements, radius)
        trees = struct('thresholds', [], 'depth', 0, 'nOutputs', 0);
        features = zeros(nAug, numFeatsTotal, 'int8');
        
        % Generate regression trees & extract features
        offsetFeats = 0;
        fprintf('Generating trees: ');
        for iL = 1:numLandmarks
            % Get differences between ground truth and current landmark locaions
            clmkDisplacements = reshape(lmkDisplacements(iL,:,:), 2, [])';
            for iT = 1:numLmkTrees
                [trees(numLmkTrees*(iL-1)+iT), features(:,(1:numFeatsTree)+offsetFeats)] = generateRegressionTree( ...
                    iL, ...
                    transforms, ...
                    clmkDisplacements, ...
                    radius ...
                );
                offsetFeats = offsetFeats + numFeatsTree;
                fprintf('.');
            end
        end
        disp(' ');
    end
    
    function [tree, results] = generateRegressionTree(iLmk, transforms, dispsLandmark, radii)
        function [nodes] = splitData(indexs, lvls, nodes, tnode)
            if length(indexs) < 31*2.5^(lvls-1)
                nodes = [];
                return;
            end
            
            [coords,pixels] = samplePixels(iLmk, radii, indexs, transforms);
            [p1,p2,~]       = splitMinimizingTargetVariance(pixels, dispsLandmark(indexs,:));
            nodes           = [nodes 0 reshape(coords([p1 p2],:)', 1, [])];
            inds            = pixels(:,p1) < pixels(:,p2);
            
            if lvls > 1
                nodes = splitData(indexs(inds),  lvls-1, nodes, tnode);
                if isempty(nodes), return; end
                nodes = splitData(indexs(~inds), lvls-1, nodes, tnode+2^(lvls-1));
                if isempty(nodes), return; end
            else
                results(indexs(inds),  tnode)   = 1;
                results(indexs(~inds), tnode+1) = 1;
            end
        end
        
        thresholds = [];
        while isempty(thresholds)
            results    = zeros(nAug, 2^treeDepth, 'int8');
            thresholds = splitData(1:nAug, treeDepth, [], 1);
            if isempty(thresholds), fprintf('!'); end
        end
        tree = struct('thresholds', thresholds, 'depth', treeDepth, 'nOutputs', 2^treeDepth);
    end

    function [coords, pixels] = samplePixels(iLmk, radii, imInds, transforms)
        nI = length(imInds);

        % Select pool of coordinates
        theta = rand(NUM_SAMPLE_PIXELS,1,'single')*(2*pi);
        r = sqrt(rand(NUM_SAMPLE_PIXELS,1,'single'))*radii;
        coords = [r.*cos(theta) r.*sin(theta)];            
        
        % Obtain pixels from coordinates for each image
        pixels = rand(nI, NUM_SAMPLE_PIXELS, 'single');
        tcoords = round(coords*transforms);
        for iI = 1:nI
            dispW = int32(lmkEstimates(iLmk,1,imInds(iI)) + tcoords(:,(2*iI-1)));
            dispH = int32(lmkEstimates(iLmk,2,imInds(iI)) + tcoords(:,(2*iI)));
            vind  = dispH > 0 & dispH <= imH & dispW > 0 & dispW <= imW;
            pixels(iI,vind) = images((dMap(imInds(iI))-1)*(imW*imH) + dispH(vind) + (dispW(vind)-1)*imH);
        end
    end

    %% --------------------------------------------------------------------
    %  -- AUGMENT TRAINING DATA
    %  --------------------------------------------------------------------

    nAug = int32(NUM_AUGMENTS*numImg);
    dMap = 1 + floor(((0:nAug-1)-mod((0:nAug-1),NUM_AUGMENTS)) / NUM_AUGMENTS);
    
    % Prepare target shapes
    groundTruth = groundTruth(:,:,dMap);
    
    % Prepare initial shape estimates
    meanShape = generalizedProcrustes2D(groundTruth)';
    meanOff = mean(meanShape, 1);
    meanSize = max(meanShape, [], 1) - min(meanShape, [], 1);
    
    if ~estimatesSet
        lmkEstimates = zeros(numLandmarks, 2, nAug, 'single');
        for i = 1:nAug
            rAngle = rand('single')*pi - pi/2;
            rScale = 0.75 + rand('single')*0.5;
            rC = cos(rAngle); rS = sin(rAngle);

            lmkEstimates(:,:,i) = bsxfun(@plus, bsxfun(@minus, meanShape, meanOff) * (rScale * [rC rS ; -rS rC]), meanOff); % Rotation + scaling
            lmkEstimates(:,:,i) = bsxfun(@plus, lmkEstimates(:,:,i), meanSize .* (rand(1,2)-0.5)*0.4); % Displacement
        end
    end
    
    % Calculate pseudo-inverse of mean shape
    pinvMeanShape = pinv([meanShape ones(numLandmarks, 1, 'single')]);
    
    %% --------------------------------------------------------------------
    %  -- GENERAL TRAINING CODE
    %  --------------------------------------------------------------------
    
    % Prepare return structure
    if ~exist('shapeRegressor', 'var')
        % If doesn't exist, create structure
        shapeRegressor = struct( ...
            'meanShape', meanShape, ...
            'numStageTrees', numTrees, ...
            'treeDepth', treeDepth, ...
            'stages', repmat(struct( ...
                'trees', single([]), ...
                'weights', single([]) ...
            ), [1, numStages]) ...
        );
    elseif ~estimatesSet
        % If exists and estimates not provided, regress current shapes
        lmkEstimates = fastFacealign_test(shapeRegressor, images, 'initialShapes', lmkEstimates, 'shapeImages', dMap);
        shapeRegressor.stages(end+1:numStages) = struct('trees', [], 'weights', []);
    end

    shapeRegressor.stages(end+1:numStages) = struct('trees', [], 'weights', []);
    for iS = 1:numStages
        if ~isempty(shapeRegressor.stages(iS).trees), continue; end
        [shapeRegressor.stages(iS),lmkUpdates] = generateStage(groundTruth, lmkEstimates);
        lmkEstimates = lmkEstimates + permute(reshape(lmkUpdates', 2, numLandmarks, []), [2 1 3]);
        save('tmpModel_tmp.mat', 'shapeRegressor', 'lmkEstimates');
    end
    
    save('tmpModel.mat', 'shapeRegressor');
end