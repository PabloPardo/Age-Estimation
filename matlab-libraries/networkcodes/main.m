%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THIS CODE IMPLEMENTS THE 4 LAYER ARCHITECTURE FOR IMAGE CLASSIFICATION
% DESCRIBED IN THE FOLLOWING ARTICLES:
%
% EXTENDED CODING AND POOLING IN THE HMAX MODEL.
% CHRISTIAN THERIAULT,NICOLAS THOME AND MATTHIEU CORD. TO APPEAR IN IEEE
% TRANSACTIONS ON IMAGE PROCESSING 2013 (ALREADY ONLINE)
%
% HMAX-S: DEEP SCALE REPRESENTATION FOR BIOLOGICALLY INSPIRED IMAGE CATEGORIZATION
% CHRISTIAN THERIAULT, NICOLAS THOME, MATTHIEU CORD. ICIP 2011, P 1261-1264,
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Indicate the path where the image set is found. This folder should contain a subfolder for each image category.
ImageSet='/.../.../101_ObjectCategories/';

% path where MinMaxFilterFolder is found. This function implements local maxima selection,
maxpath='/.../.../networkcodes/MinMaxFilterFolder/MinMaxFilterFunctions/';

% Indicate the path where networkcodes folder is found
codespath='/.../.../networkcodes/';

% Set Matlab paths
path(path,maxpath);
path(path,codespath);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GET CATEGORY NAMES, THE IMAGES PATH AND THE LABELS
[catnames labels image_paths] =get_file_paths(ImageSet);
save('labels','labels')
%% GABOR FILTERS PARAMETERS
% These are the 16 scales, 16 wavelengths and 16 Gabor filter sizes  (see the READ_ME.txt for a reference paper on these values)
% These lists maybe modified
Pscale=[2.0 2.8 3.6 4.5 5.4 6.3 7.3 8.2 9.2 10.2 11.3 12.3 13.4 14.6 15.8 17.0];
Pwavelength=[2.5 3.5 4.6 5.6 6.8 7.9 9.1 10.3 11.5 12.7 14.1 15.4 16.8 18.2 19.7 21.2];
Pfiltersize=[5:2:35];
%% LAYERS PARAMETERS AND INITIALIZATIONS
Number_CPUs=4;                                       % Number of CPUs on machine: maximum = 8
NumberofCat=length(catnames);                        % Number of image categories

% Image paramaters
shortestside=140;                                    % Images are rescaled so the shortest is equal to "shortestside" keeping aspect ratio

% Layer 1 parameters
NumbofOrient=12;                                    % Number of spatial orientations for the Gabor filter on the first layer
Numberofscales=16;                                   % Number of scales for the Gabor filter on the first laye: must be between 1 and 16.
% Modify line 7-9 of create_gabors.m to increase to more than than 16 scales
% Layer 2 layer parameters
maxneigh=floor(8:length(Pscale)/Numberofscales:8+length(Pscale));  % Size of maximum filters (if necessary adjust according Gabor filter sizes)
L2stepsize=4;                                                      % Step size of L2 max filter (downsampling)
inhi=0.5;                                                          % Local inhibition ceofficient (competition) between Gabor outputs at each position.
% Coefficient is between 0 and 1
% Layer 3 (Learning parameters)
NumOfSampledImages=40;                               % Number of images per category from which the dictionary is learned
featureperimage=1;                                   % Number of features learned from each image
feature_spatial_sizes=[4:4:16];                      % List of possible spatial sizes for features: Maximum value must be less than (shortestside/L2stepsize)
feature_scaledepth=7;                                % Scale depth of dictionary features (i.e L3 filters): Must be less than Numberofscales

% Layer 4 paramaters
pooling_radii=[0.12];                                % concentric spatial pooling radii at layer L4 (image percentage)
% Example for multi-resolution pooling_radii= [0.05 0.12 0.3 0.5 0.7 1.0]; (SLOWER)

%% LOAD AND DISPLAY GABOR FILTERS
Gabor=create_gabors(Numberofscales,NumbofOrient,Pscale,Pwavelength,Pfiltersize);
displaygabors(Gabor)

%%  GENERATE DICTIONARY
%
% % Initialize
numFeatures =NumberofCat*NumOfSampledImages*featureperimage;     % Total number of features in the dictionary.
featurecounter=0;
dictionary=cell(numFeatures,1);
feature_sizes=zeros(numFeatures,1);                              % List containing the size of each feature
Scale_and_Position=zeros(numFeatures,5);                         % List containing the position, the scale and the layer size at which
% each feature is sampled.
%%
fprintf('CREATING DICTIONARY BY SAMPLING FROM TRAINING IMAGES\n');
imagecounter=0;                                                     % counter for the images in the current category.
try
    for (cat=1:NumberofCat)                                             % Loop over the categories .
        
        category_feature_counter=0;                                         % counter for feature in the current category.
        ima=0;
        notreach=true;
        while (category_feature_counter~=NumOfSampledImages)                % Loop over the number of sampled images for this category.
            
            category_feature_counter=category_feature_counter+1;
            if(ima<length(find(labels==cat))&&notreach)
                ima=category_feature_counter;
            else
                notreach=false;
                ima=mod(category_feature_counter,length(find(labels==cat)));
            end
            imagecounter=imagecounter+1;
            
            fprintf('sampling %u feature(s) from %s\n', featureperimage, image_paths{find(labels==cat,1,'first')+ima-1});
            
            % LOAD IMAGE
            A=imread(image_paths{find(labels==cat,1,'first')+ima-1});
            A = mean(A,3);                                                              % convert to grayscale.
            [m n]=size(A);
            if (m<n)
                A=imresize(A,[shortestside NaN]);                                           % resize image.
            else
                A=imresize(A,[NaN shortestside]);
            end
            [m n]=size(A);
            
            
            % L1 LAYER  (NORMALIZED DOT PRODUCT OF GABORS FILTERS ON LOCAL PATCHES OF IMAGE "A" AT EVERY POSSIBLE LOCATIONS AND SCALES)
            L1 = L1_layer(Gabor, A);
            
            % L2 LAYER: LOCAL MAX POOLING OF L1 LAYER OVER LOCAL POSITION AT ALL SCALES AND ALL ORIENTATIONS
            % THE MAXIMUM POOLING SLIDING WINDOW SIZES ARE CONTAINED IN "maxneigh" AND "L2stepsize" INDICATES THE CORRESPONDING STEPSIZE
            L2 = L2_layer(L1,L2stepsize,maxneigh);
            
            % EXTRACT AND STORE FEATURE
            [a b c d]=size(L2);
            
            for(p=1:featureperimage)                                     % Loop over the sampled features for the current image.
                
                [feature featuresize s i j] = sample_from_L2(L2,feature_spatial_sizes,feature_scaledepth);   % extract a feature at random position on layer L2.
                % The feautre has a scale depth equal to
                % feature_scaledepth and a spatial size randomly
                % selected in feature_spatial_sizes
                featurecounter=featurecounter+1;
                dictionary{featurecounter}=feature;                          % the dictionary of feature.
                feature_sizes(featurecounter)=featuresize;                   % the spatial size of each feature.
                Scale_and_Position(featurecounter,:)=[i,j,s,a,b];            % the sampled scale (s), the sampled spatial position (i,j).
                % and the size (a,b) of layer L2 for this feature (used for interpolation).
            end
        end
    end
    
catch err
    if(feature_scaledepth>=Numberofscales)
        error('feature_scaledepth must be smaller than Numberofscales');
    end
    if(max(feature_spatial_sizes)>=shortestside / L2stepsize - 1)
        error('maximum of feature_spatial_sizes must be smaller than (shortestside / L2stepsize) - 1');
    end
    rethrow(err);
end

save('dictionary', 'dictionary');
save('feature_sizes', 'feature_sizes');
save('Scale_and_Position','Scale_and_Position');

%%
load('dictionary', 'dictionary');
load('feature_sizes', 'feature_sizes');
load('Scale_and_Position','Scale_and_Position');
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CREATE IMAGE SIGNATURES
% open matlabpool according to the specified number of cpus (used with the parfoor loop command below)
open_matlabpool(Number_CPUs)

% initialization
all_images_signatures=zeros(length(pooling_radii)*numFeatures,length(image_paths));
dictionary_norm=cell(numFeatures,1);
% normalize and flip filters (dictionary elements)
for (i=1:numFeatures)
    dictionary_norm{i}=dictionary{i}/norm(dictionary{i}(:));
    dictionary_norm{i}(:,:,:,:)=dictionary_norm{i}(end:-1:1,end:-1:1,:,:);
end


% Begin image signature creation

fprintf('Generate all images signatures   \n');
try
    for (ima=1:length(image_paths))   %
        
        fprintf('computing L4 vector for %s\n', image_paths{ima});
        
        % LOAD IMAGE
        A=imread(image_paths{ima});
        A = mean(A,3);                                                              % convert to grayscale
        [m n]=size(A);
        if (m<n)
            A=imresize(A,[shortestside NaN]);
        else
            A=imresize(A,[NaN shortestside]);
        end
        [m n]=size(A);
        
        
        % L1 LAYER  (NORMALIZED DOT PRODUCT OF GABORS FILTERS ON LOCAL PATCHES OF IMAGE "A" AT EVERY POSSIBLE LOCATIONS AND SCALES)
        L1 = L1_layer(Gabor, A);
        
        % L2 LAYER: LOCAL MAX POOLING OF L1 LAYER OVER LOCAL POSITION AT ALL SCALES AND ALL ORIENTATION
        % THE MAXIMUM POOLING SLIDING WINDOW SIZES ARE CONTAINED IN "maxneigh" AND "L2stepsize" INDICATES THE CORRESPONDING STEPSIZE
        L2 = L2_layer(L1,L2stepsize,maxneigh);
        [a b c d]=size(L2);
        
        
        % COMPETITION (INHIBITION) BETWEEN L2 OUTPUTS AT EACH SPATIAL POSITION ON SUBSETS OF "FEATURE_SCALEDEPTH"
        % EXAMPLE: NUMBEROFSCALES=8, FEATURE_SCALEDEPTH=5. INHIBITION IS APPLIED SEPERATLY ON SUBSETS {1,2,3,4,5},{2,3,4,5,6},{3,4,5,6,7},{4,5,6,7,8}
        L2new=cell(Numberofscales,1);
        for (s=(feature_scaledepth+1)/2:Numberofscales-(feature_scaledepth+1)/2)
            tmp=L2(:,:,s-(feature_scaledepth-1)/2:s+(feature_scaledepth-1)/2,:);
            [a b c z]=size(tmp);
            ma=max(max(tmp,[],4),[],3);
            mi=min(min(tmp,[],4),[],3);
            thres=mi+inhi*(ma-mi);
            L2new{s}=tmp.*(tmp>reshape(repmat(thres,1,NumbofOrient*feature_scaledepth),a,b,feature_scaledepth,NumbofOrient));  % local inhibition on scale global on orientation
        end
        
        % LAYER L3 + LAYER L4
        % L3: MAPS THE DICITONARY FEATURES AT EVERY POSITION OF THE L2 LAYER AND AT +- 1 SCALE CENTERED ON THE SCALE AT WHICH THE FEATURE WAS SAMPLED.
        % L4: POOL THE MAXIMUM L3 ACROSS ALL L3 SCALES AND OVER CONCENTRIC SPATIAL RADII.
        
        L4=zeros(numFeatures,length(pooling_radii));
        parfor (p=1:numFeatures)   % Loop over all feature in the dictionary
            
            L2new_2=L2new;         % rename for parfoor loop
            maxneigh_2=maxneigh;
            Scale_and_Position_2=Scale_and_Position;
            
            L3=cell(3,1);
            centerscale=Scale_and_Position_2(p,3);                                  % scale at which the feautre was sampled when creating the dictionary
            lowerscale=max((feature_scaledepth+1)/2,centerscale-1);                 % lower pooling scale
            upperscale=min(Numberofscales-(feature_scaledepth+1)/2,centerscale+1);  % upper pooling scale
            
            % L3: maps the dicitonary features at every position of the L2 layer and at +- 1 scale centered on the scale at which the feature was sampled.
            ct=0;
            for (s=lowerscale:upperscale)                               % loop on scales
                ct=ct+1;
                ff=convnd_2_norm(L2new_2{s},dictionary_norm{p});
                L3{ct}= ff;
            end
            
            % L4 layer : pool the maximum over concentric spatial radii and across all L3 scales
            
            radii_signature=zeros(length(pooling_radii),1);
            
            for (rd=1:length(pooling_radii))    % loop on pooling radius
                radius=pooling_radii(rd);
                
                % Retreive the spatial coordinate at which the feature was sampled
                y=Scale_and_Position_2(p,1);
                x=Scale_and_Position_2(p,2);
                u=Scale_and_Position_2(p,4);
                v=Scale_and_Position_2(p,5);
                [a b c d]=size(L2);
                nx=ceil(intervalmap(1,v,1,b,x));            % interpolate position to account for the different size of the current
                ny=ceil(intervalmap(1,u,1,a,y));            % image and the image on which the feature was sampled.
                [d e f z]=size(L1);
                
                % For each L3 scale pool the maximum value inside each radius
                sc=0;
                ma=zeros(3,1);
                for (s=lowerscale:upperscale)                                               % loop on pooling scales
                    sc=sc+1;
                    yrange=ceil(radius*d);                                                      % pooling radius in terms of percentage of layer L1 size
                    xrange=ceil(radius*e);
                    yrange=max(0,ceil(intervalmap(maxneigh(s),maxneigh(s)+(a-1)*L2stepsize,1,a,yrange)));  % interpolate the pooling radius to corresponding L2 layer size
                    xrange=max(0,ceil(intervalmap(maxneigh(s),maxneigh(s)+(b-1)*L2stepsize,1,b,xrange)));
                    [g h]=size(L3{sc});
                    x1=min(h,max(1,nx-xrange));             % pooling spatial limits
                    x2=min(h,nx+xrange);
                    y1=min(g,max(1,ny-yrange));
                    y2=min(g,ny+yrange);
                    ma(sc)=max(max(L3{sc}(y1:y2,x1:x2)));   % maximum inside pooling window
                end
                % pool over scale
                radii_signature(rd)=max(ma);
            end
            L4(p,:)=radii_signature; % update signature for this feature
        end
        all_images_signatures(:,ima)=L4(:);  % signature for this image
        clear L4
    end
    
catch err
    matlabpool close
    if(feature_scaledepth>=Numberofscales)
        error('feature_scaledepth must be smaller than Numberofscales');
    end
    if(max(feature_spatial_sizes)>=shortestside / L2stepsize - 1)
        error('maximum of feature_spatial_sizes must be smaller than shortestside / L2stepsize - 1');
    end
    rethrow(err);
end

save('all_images_signatures','all_images_signatures','-v7.3');
%%
addpath(genpath('/home/theriaultc/liblinear-1.91/'));
addpath(genpath('/home/theriaultc/libsvm-3.12/'));
addpath(genpath('/home/theriaultc/use_libsvm/'));
%%
load('all_images_signatures','all_images_signatures');
%% creating training /testing split
load('labels','labels')
signatures=all_images_signatures';
number_of_train_example=30;
max_number_of_test=50;

rdm=randperm(length(labels));
%load('rdm','rdm')
labels=labels(rdm);
signatures=signatures(rdm,:);
%%

trainid=[];
for (i=1:max(labels))
    trainid=[trainid find(labels==i,number_of_train_example,'first') ];
end

catsize=zeros(102,1);
for (i=1:102)
    catsize(i)=length(find(labels==i));
end

testid=[];
for (i=1:102)
    if (catsize(i)<number_of_train_example+max_number_of_test)
        testid=[testid find(labels==i,catsize(i)-number_of_train_example,'last') ];
    else
        testid=[testid find(labels==i,max_number_of_test,'last') ];
    end
end

%save('rdm','rdm')



%% Classification: One vs all
%puts data into libsvm format

train_label_vector = labels(trainid)';
trainfeatures = signatures(trainid, :);

test_label_vector = labels(testid)';
testfeatures = signatures(testid, :);

numTrain = size(trainfeatures,1);
numTest = size(testfeatures,1);

%% Train one SVM per class
numLabels=max(train_label_vector);
% %# train one-against-all models
model = cell(numLabels,1);
for k=1:numLabels
    model{k} = svmtrain(double(train_label_vector==k), trainfeatures, '-t 0 -c 1 -b 1');
end
save('model','model')
%% Test: get probability estimates of test instances using each SVM
prob = zeros(numTest,numLabels);
for k=1:numLabels
    [~,~,p] = svmpredict(double(test_label_vector==k), testfeatures, model{k}, '-b 1');
    prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
end
save('prob','prob')
%% predict the class with the highest probability
[~,pred] = max(prob,[],2);
acc = sum(pred == test_label_vector) ./ numel(test_label_vector)    %# accuracy