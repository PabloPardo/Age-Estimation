function [shapes] = fastFacealign_test(model, images, varargin)
    addpath(genpath('./libs'));
    
    %% --------------------------------------------------------------------
    %  -- LOAD ALGORITHM PARAMETERS
    %  --------------------------------------------------------------------
    
    for i = 1:2:length(varargin)
        k = varargin{i};
        v = varargin{i+1};
        if strcmp(k, 'initialShapes')
            shapes         = v;
        elseif strcmp(k, 'shapeImages')
            dMap           = v;
            nAug           = length(dMap); 
        end
    end
    
    [imH,imW,nI] = size(images);
    treeDepth    = model.treeDepth;
    treeFeats    = 2^treeDepth;
    nS           = length(model.stages);
    nT           = model.numStageTrees;
    nL           = size(model.meanShape, 1);
    nLT          = ceil(nT/nL);
    
    %% --------------------------------------------------------------------
    %  -- STAGES EXECUTION METHODS
    %  --------------------------------------------------------------------
    
    % Tree feature extraction
    function [features] = extractFeaturesFromTree(tree, transforms, offsets)
        function [features] = visitLevel(treeData, imInds, lvl)
            % Obtain pixel differences
            nImg = length(imInds);
            pixCoords = reshape(treeData(2:5), 2, 2)' * reshape(transforms(:,imInds), 2, []);
            pixDiffs  = zeros(1,nImg);
            
            dispW = round(bsxfun(@plus, pixCoords(:,(1:nImg)*2-1), offsets(1,imInds)));
            dispH = round(bsxfun(@plus, pixCoords(:,(1:nImg)*2-0), offsets(2,imInds)));
            vind  = dispH > 0 & dispH <= imH & dispW > 0 & dispW <= imW;
            if sum(vind(1,:)) > 0
                pixDiffs(1,vind(1,:)) = images((dMap(imInds(vind(1,:)))-1)*(imW*imH) + dispH(1,vind(1,:)) + (dispW(1,vind(1,:))-1)*imH);
            end
            if sum(vind(2,:)) > 0
                pixDiffs(1,vind(2,:)) = pixDiffs(1,vind(2,:)) - images((dMap(imInds(vind(2,:)))-1)*(imW*imH) + dispH(2,vind(2,:)) + (dispW(2,vind(2,:))-1)*imH);
            end
            
            % Split into subnodes
            indSplits = pixDiffs < 0;
            if lvl < treeDepth
                subSize = 5*(2^(treeDepth-lvl)-1);
                features = [ ...
                    visitLevel(treeData(6:5+subSize), imInds(indSplits), lvl+1), ...
                    visitLevel(treeData(6+subSize:5+2*subSize), imInds(~indSplits), lvl+1) ...
                ];
            else
                features = zeros(nAug, 2);
                features(imInds(indSplits),  1) = 1;
                features(imInds(~indSplits), 2) = 1;
            end
        end
        
        features = visitLevel(tree.thresholds, 1:nAug, 1);
    end
    
    % Feature extraction
    function [features] = extractFeatures(trees)
        % Extract features
        features = zeros(nAug,nLT*nL*treeFeats);
        for iT=1:nLT*nL
            lmkInd = ceil(double(iT) / double(nLT));
            features(:,(1+(iT-1)*treeFeats):(iT*treeFeats)) = extractFeaturesFromTree( ...
                trees(iT), transforms, reshape(shapes(lmkInd,:,:), 2,[]) ...
            );
        end
    end
    
    %% --------------------------------------------------------------------
    %  -- GENERAL TESTING CODE
    %  --------------------------------------------------------------------

    % Pre-calculate pseudo-inverse of the mean shape
    pinvMeanShape = pinv([model.meanShape ones(nL,1)]);

    % Initialize shapes
    if ~exist('shapes', 'var')
        shapes = repmat(model.meanShape, [1 1 nI]);
        nAug   = nI;
        dMap   = 1:nI;
    end
    
    % Regress shapes
    for iS = 1:nS
        % Define transforms from mean shape to current shape estimates
        transforms = zeros(4, nAug);
        for iI = 1:nAug
            tfm = pinvMeanShape * shapes(:,:,iI);
            transforms(:,iI) = reshape(tfm(1:2,:), 4, 1);
        end
        
        % Extract features and update shape estimates
        features = extractFeatures(model.stages(iS).trees);
        shapeUpdates = permute(reshape((features * model.stages(iS).weights)', 2, nL, []), [2 1 3]);
        for iI = 1:nAug
            shapes(:,:,iI) = shapes(:,:,iI) + shapeUpdates(:,:,iI) * reshape(transforms(:,iI), 2, 2);
        end
    end
end