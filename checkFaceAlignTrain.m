function [shape] = checkFaceAlignTrain(model, images, varargin)

for i = 1:2:length(varargin)
        k = varargin{i};
        v = varargin{i+1};
        if strcmp(k, 'initialShapes')
            initialShapes         = v;
        end
end

%% Initialize with different shapes
[n, m, nInit] = size(initialShapes);
nInst = size(images, 3);
shapes = zeros(n, m, nInit, nInst);
myCluster=parcluster('local'); 
myCluster.NumWorkers=4; 
parpool(myCluster,4)
parfor i = 1:nInst
    disp(sprintf('%i / %i\n', i, nInst));
    shapes(:,:,:,i) = fastFacealign_test(model, double(images(:,:,i))/255, ...
        'initialShapes', initialShapes, 'shapeImages', ones(1, nInit));
end

%% Find Centroid
shape = mean(shapes, 3);
