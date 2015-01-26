addpath(genpath('matlab-libraries/3000fps - custom'));

%% CREATE FACE DETECTORS
% disp('Creating face detectors ...');
% import vision.*;
% detector1 = vision.CascadeObjectDetector('data/cascade.xml');
% detector2 = vision.CascadeObjectDetector();

%% LOAD SHAPE MODEL
disp('Loading shape model ...');
load('data/model68.mat');

%% READ DATA
disp('Reading data ...');
% HuPBA database
path = '../../Databases/Aging DB/AGE HuPBA/';
fileID = fopen([path 'HuPBA_AGE_data_extended.csv']);

% FGNET database
% path = '../../Databases/Aging DB/FGNET/';
% fileID = fopen([path 'ages.csv']);

% if exist('shapes.mat', 'file') == 2
%     load('shapes.mat');                         % load data
if exist('data/FGNssssET_shapes.mat', 'file') == 2
    load('data/FGNET_shapes.mat'); 
    BLOCK_SIZE = 2000;                          % initial capacity (& increment size)
    listSize = size(shapes,3);                  % current list capacity
    listPtr = size(shapes,3);                   % pointer to last free position
%     Y = zeros(listSize);                        % Target
    Y = zeros(listSize, 2);                     % Target - HuPBA AGE
else
    BLOCK_SIZE = 2000;                          % initial capacity (& increment size)
    listSize = BLOCK_SIZE;                      % current list capacity
    shapes = zeros(68,2,listSize);              % Shapes (landmarks)
%     Y = zeros(listSize);                        % Target
    Y = zeros(listSize, 2);                     % Target - HuPBA AGE
    listPtr = 1;                                % pointer to last free position
end

margin = 20;    % Margin of the faces

line = fgetl(fileID);
line = fgetl(fileID);
while line~=-1
    timerVal = tic();
    c = strsplit(line, ',');
    
    % Store the age
%     Y(listPtr) = str2double(c{2});      % FERET
%     Y(listPtr) = str2double(c{1});      % FG-NET
    Y(listPtr, 1) = str2double(c{4});   % HuPBA AGE - Real Age
    Y(listPtr, 2) = str2double(c{5});   % HuPBA AGE - Apparet Age
    
    % Read the image
    if exist(['images/face_detect/' c{3}], 'file') == 2
%         img = imread(c{3});                     % FERET 
%         img = imread([path 'all_img/' c{3}]);   % FG-NET
        img = imread(['images/face_detect/' c{3}]);  % HuPBA AGE
        disp(c{3});
    else
        disp('Not finded');
        line = fgetl(fileID);
        toc(timerVal)
        continue;
    end
    
    %% Detect face in the image and crop
%     face = detectface(img, margin, detector1, detector2);
%     [img, shape] = FGNETface(img, [path 'all_img/' c{3}]);
%     if isempty(face)
%         line = fgetl(fileID);
%         continue;
%     end
        
    % Crop the image by the found coordinates
%     img = imresize(imcrop(img, face), [200,200]);
    img = imresize(img, [200,200]);
    
    %% Find facial landmarks
    shape = fastFacealign_test(shapeRegressor, double(img)/255);
    
    %% Align Face
    [rot_image, rot_shape] = alignface(img, 42, 48, 31, shape(:,:,1));     % FERET & HuPBA AGE
%     [rot_image, rot_shape] = alignface(img, 32, 37, 68, shape);   % FG-NET
    shapes(:,:,listPtr) = rot_shape;

    %% Save image
%     imwrite(rot_image, strcat('../', strrep(c{3}, 'Original', 'Aligned')), 'jpg');  % FERET
%     imwrite(rot_image, strcat(path, 'all_images_aligned/', c{3}), 'jpg');           % FG-NET
    imwrite(rot_image, [path 'extended_aligned/' c{3}], 'png');                     % HuPBA AGE

    % add new block of memory if needed
    listPtr = listPtr + 1;
    if( listPtr+(BLOCK_SIZE/100) > listSize )   % less than 1%*BLOCK_SIZE free slots
        listSize = listSize + BLOCK_SIZE;       % add new BLOCK_SIZE slots
%         Y(listPtr+1:listSize) = 0;            % FERET & FG-NET
        Y(listPtr+1:listSize,:) = 0;            % HuPBA AGE
        shapes(:,:,listPtr+1:listSize) = 0;
%         save('data/shapes', 'shapes');          % FERET
%         save('data/FG-NET/FGNET_shapes', 'shapes');    % FG-NET
%         save('data/FG-NET/FGNET_Y', 'Y');              % FG-NET
        save('data/HuPBA/HuPBA_shapes', 'shapes');    % HuPBA AGE  
        save('data/HuPBA/HuPBA_Y', 'Y');              % HuPBA AGE
    end
  
    line = fgetl(fileID);
    toc(timerVal)
end
fclose(fileID);

% remove unused slots
Y(listPtr:end,:) = [];
shapes(:,:,listPtr:end) = [];

% save('data/Y', 'Y');                    % FERET
% save('data/shapes', 'shapes');          % FERET
% save('data/FG-NET/FGNET_Y', 'Y');              % FG-NET
% save('data/FG-NET/FGNET_shapes', 'shapes');    % FG-NET
save('data/HuPBA/HuPBA_shapes', 'shapes');    % HuPBA AGE  
save('data/HuPBA/HuPBA_Y', 'Y');              % HuPBA AGE