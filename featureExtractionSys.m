disp('Loading paths ...');
% addpath('data/');
addpath(genpath('../Train/'), genpath('matlab-libraries/networkcodes'));

%% SET BIF PARAMETERS
disp('Setting parameeters ...');
param = struct;
% GABOR FILTERS PARAMETERS
% These are the 16 scales, 16 wavelengths and 16 Gabor filter sizes  (see the READ_ME.txt for a reference paper on these values)
% These lists maybe modified
param.Pscale = [2.0 2.8 3.6 4.5 5.4 6.3 7.3 8.2 9.2 10.2 11.3 12.3 13.4 14.6 15.8 17.0];
param.Pwavelength = [2.5 3.5 4.6 5.6 6.8 7.9 9.1 10.3 11.5 12.7 14.1 15.4 16.8 18.2 19.7 21.2];
param.Pfiltersize = [5:2:35];
% LAYERS PARAMETERS AND INITIALIZATIONS
% Image paramaters
param.shortestside = 200;                    % Images are rescaled so the shortest is equal to "shortestside" keeping aspect ratio

% Layer 1 parameters
param.NumbofOrient = 12;                     % Number of spatial orientations for the Gabor filter on the first layer
param.Numberofscales = 16;                   % Number of scales for the Gabor filter on the first laye: must be between 1 and 16.
% Modify line 7-9 of create_gabors.m to increase to more than than 16 scales
% Layer 2 layer parameters
param.Ns = 10;                                % Number of grid divisions Ns x Ns of the L1 to compute the STD
param.maxneigh = floor(4:length(param.Pscale)/param.Numberofscales:4+length(param.Pscale));  % Size of maximum filters (if necessary adjust according Gabor filter sizes)
param.L2stepsize = 4;                        % Step size of L2 max filter (downsampling)
param.inhi = 0.5;                            % Local inhibition ceofficient (competition) between Gabor outputs at each position.
% Coefficient is between 0 and 1

%% READ DATA
disp('Reading data ...');
% path = '../Train/Aligned/';
% path = '../../Databases/Aging DB/FGNET/';
path = '../../Databases/Aging DB/AGE HuPBA/';
% fileID = fopen(strcat(path, 'aligned.csv'));
% fileID = fopen(strcat(path, 'ages.csv'));
fileID = fopen(strcat(path, 'HuPBA_AGE_data_extended.csv'));

disp('Starting Feature Extraction ...');
features_size = round((2*param.Ns-1)/param.L2stepsize) * ...
                round(param.Ns/param.L2stepsize) * ... 
                param.NumbofOrient * (param.Numberofscales-1);
BLOCK_SIZE = 2000;                          % initial capacity (& increment size)
listSize = BLOCK_SIZE;                      % current list capacity
X = zeros(features_size,listSize);          % BIF Features
% Y = zeros(listSize,1);                      % Targets
listPtr = 1;                                % pointer to last free position

margin = 20;    % Margin of the faces

% line = fgetl(fileID);
line = fgetl(fileID);
while line~=-1
    timerVal = tic();
    c = strsplit(line, ',');
    
%     Y(listPtr) = str2double(c{1});
%     img = imread([path 'all_images_aligned/' c{3}]); % Read the image
    img = imread([path 'extended_aligned/' c{2}]);
    
    %% Find BIF features
    X(:,listPtr) = BIFextractor(img, param);
    
    % add new block of memory if needed
    listPtr = listPtr + 1;
    if( listPtr+(BLOCK_SIZE/100) > listSize )  % less than 1%*BLOCK_SIZE free slots
        X(:,listPtr:end,:) = [];
        Y(listPtr:end) = [];
%         save('data/FGNET_X', 'X');
%         save('data/X', 'X');
%         save('data/Y', 'Y');
        save('data/HuPBA_X', 'X');
        
        listSize = listSize + BLOCK_SIZE;       % add new BLOCK_SIZE slots
        X(:,listPtr+1:listSize) = 0;
%         Y(listPtr+1:listSize) = 0;
    end
  
    line = fgetl(fileID);
    fprintf('Time: %2.2f iter: %d\n', toc(timerVal), listPtr-1);
end
fclose(fileID);

% remove unused slots
X(:,listPtr:end,:) = [];
% Y(listPtr:end) = [];

% save('data/FGNET_X', 'X');
% save('data/X', 'X');
% save('data/Y', 'Y');
save('data/HuPBA_X', 'X');