clc; clear all; close all;
addpath(genpath('matlab-libraries/libsvm-3.18/matlab'));

% Load Data
load('data/FGNET_X.mat');
load('data/FGNET_Y.mat');
load('data/FGNET_shapes.mat')
csvwrite('data/FGNET_shapes.csv', reshape(shapes, [size(shapes,1)*size(shapes,2)], size(shapes,3)));
csvwrite('data/FGNET_X.csv', X);
csvwrite('data/FGNET_Y.csv', Y);

% Add shape coodinates to X features matrix
shapes_vec = [];
size_X = size(X,1);
X(size_X+1:size_X+136,:) = 0;
for i = 1:size(shapes,3)
    aux = shapes(:,:,i);
    shapes_vec = aux(:);
    X(size_X+1:size_X+136,i) = shapes_vec;
end
csvwrite('data/X_shapes.csv', X);

% Normalize the data
X = X';
X = (X-repmat(mean(X,2),[1,size(X,2)]))./repmat(std(X,1),[size(X,1),1]);

N = size(X, 2); % Number of samples
k = 25;         % Number of features to keep

%% PCA
% Get the k most important components
[~,SCORE,latent] = princomp(X);
proj_X = SCORE(:,1:k);
disp(sum(latent(1:k))/sum(latent));

% Normalize the data
proj_X = (proj_X-repmat(mean(proj_X,2),[1,size(proj_X,2)]))./repmat(std(proj_X,1),[size(proj_X,1),1]);

%% Cross Validation
[Test, Train] = crossvalind('HoldOut', N, 0.75);

Train_set_X = proj_X(Train,:);
Train_set_Y = Y(Train);
Test_set_X = proj_X(Test,:);
Test_set_Y = Y(Test);

% Create mesh of parameters
kernel = [0,2];                                     % Kernel (linear and rbf)
%gamma = [1/50,1/10,1/5,1/2,1,1.5,2,5,10];           % Gamma of the RBF kernel
gamma = [100, 10, 1]; 
%cost = [1/2,1,2,8,10,50,100];                       % Penalty parameter
cost = [0.001, 100];  
epsilon = [0.001, 0.01, 0.1];
% nu = 0.1:0.1:1;                                     % Upper bound for the 
                                                    %  training errors and under 
                                                    %  bound for the no. suppot 
                                                    %  vectors
[mesh_g, mesh_c, mesh_e] = meshgrid(gamma, cost, epsilon);
mesh_g = mesh_g(:); 
mesh_c = mesh_c(:);
mesh_e = mesh_e(:);

K = 5;                                         % K fold CV
n = size(Train_set_X, 1);                      % Size of Training set
Idx = crossvalind('Kfold', n, K);              % Indeces of the test set

best_g = 0; 
best_c = 0;
best_e = 0;
gb_best_err = inf;
for p = 1:length(mesh_g)
    errors = zeros([1,K]);
    time = tic();
    for f = 1:K
        % Split data into train and validation
        train_X = Train_set_X(Idx~=f,:);
        train_Y = Train_set_Y(Idx~=f);
        valid_X = Train_set_X(Idx==f,:);
        valid_Y = Train_set_Y(Idx==f);

        % Select training parameters
        param = ['-s 4 -t 2 -g ' num2str(mesh_g(p)) ' -c ' num2str(mesh_c(p)) ' -e ' num2str(mesh_e(p)) ' -q'];

        % Train SVR
        svrobj = svmtrain(train_Y, train_X, param);
        
        prediction = svmpredict(valid_Y, valid_X, svrobj, '-q');
        errors(f) = mean(abs(valid_Y - prediction)); 
    end
    time = toc(time);
    fprintf('gamma=%2.1f  cost=%2.3f  epsilon=%1.3f, MAE: %3.4f, Time: %4.1f', mesh_g(p), mesh_c(p), mesh_e(p), mean(errors), time);
    fprintf('   %2.1f%%\n', p/length(mesh_g)*100);
    if mean(errors) < gb_best_err
        gb_best_err = mean(errors);
        best_g = mesh_g(p);
        best_c = mesh_c(p);
        best_e = mesh_e(p);
        fprintf('Best parameters: gamma=%2.1f  cost=%2.3f  epsilon=%1.3f\n', best_g, best_c, best_e);
    end
end

%% Test
% Train with the best parameters
param = ['-s 4 -t 2 -g ' num2str(best_g) ' -c ' num2str(best_c) ' -e ' num2str(best_e)];

% Train SVR
svrobj = svmtrain(Train_set_Y, Train_set_X, param);

% Test
prediction_test = svmpredict(Test_set_Y, Test_set_X, svrobj);

MAE = mean(abs(Test_set_Y - prediction_test));