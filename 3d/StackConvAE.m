function StackConvAE
clear;
clc;
close all;
warning off;
rand('state', 1);

% Configuration
imageDims = [28 28 1];%64 64 3
numClasses = 10;
layerNum = 2;
lambda = 1e-4; % Weight decay parameter  
poolMethod = 'max';
convMethod = {'valid' 'full'};

% Load MNIST Train
addpath(genpath('../../'));
images = loadMNISTImages('../../../DNN/Dataset/mnist/train-images-idx3-ubyte');
images = reshape(images,imageDims(1), imageDims(2), imageDims(3),[]);%(:,1:1000)
imagesLabels = loadMNISTLabels('../../../DNN/Dataset/mnist/train-labels-idx1-ubyte');
imagesLabels(imagesLabels == 0) = 10; % Remap 0 to 10 
TrainIndex = 1 : 10;
images_train = images(:,:,:, TrainIndex);
imagesLabels_train = imagesLabels(TrainIndex);

%% STEP1 first layer para-setting and training

filterDims_Layer1 = [5 5 1];    % Filter size for conv layer
numFilters_Layer1 = 50;   % Number of filters for conv layer
poolDims_Layer1 = [2 2 1];      % Pooling dimension, (should divide imageDim-filterDim+1)
noise_Layer1 = 0.3;
poolIndex_Layer1 = getpoolIndex(imageDims, filterDims_Layer1, poolDims_Layer1, convMethod{1});

images_train_noise = images_train .* (rand(size(images_train)) > noise_Layer1);

cae1theta = caeInitParams(imageDims, filterDims_Layer1, numFilters_Layer1, poolDims_Layer1, convMethod{1});

%%SGD algorithm for training
cae1options.epochs = 2;
cae1options.minibatch = 1;
cae1options.alpha = 0.01;%1e-1
cae1options.momentum = .95;

optcae1theta = minFuncSGD(@(x, y, z) caeCost(x, y, z, filterDims_Layer1,...
                                numFilters_Layer1, poolDims_Layer1, lambda, poolIndex_Layer1, poolMethod, convMethod), ...
                                cae1theta, images_train_noise, images_train, cae1options);
                            
cae1feature = caefeedForward(optcae1theta, images_train, filterDims_Layer1, ...
    numFilters_Layer1, poolDims_Layer1, poolIndex_Layer1, poolMethod, convMethod, 0); 
cae1featureDims = (imageDims - filterDims_Layer1 + 1) ./ poolDims_Layer1;
cae1feature = reshape(cae1feature, [cae1featureDims(1:2), size(cae1feature, 1)/prod(cae1featureDims(1:2)), length(TrainIndex)]);

%% STEP2 second layer para-setting and training

filterDims_Layer2 = [5 5 numFilters_Layer1];    % Filter size for conv layer
numFilters_Layer2 = 10;   % Number of filters for conv layer
poolDims_Layer2 = [2 2 1];      % Pooling dimension, (should divide imageDim-filterDim+1)
noise_Layer2 = 0.05;
poolIndex_Layer2 = getpoolIndex([cae1featureDims(1:2) numFilters_Layer1], filterDims_Layer2, poolDims_Layer2, convMethod{1});

cae1feature_noise = cae1feature .* (rand(size(cae1feature)) > noise_Layer2);

cae2theta = caeInitParams([cae1featureDims(1:2) numFilters_Layer1], filterDims_Layer2, numFilters_Layer2, poolDims_Layer2, convMethod{1});
%%SGD algorithm for training
cae2options.epochs = 2;
cae2options.minibatch = 1;
cae2options.alpha = 0.01;%1e-1
cae2options.momentum = .95;

optcae2theta = minFuncSGD(@(x, y, z) caeCost(x, y, z, filterDims_Layer2,...
                                numFilters_Layer2, poolDims_Layer2, lambda, poolIndex_Layer2, poolMethod, convMethod), ...
                                cae2theta, cae1feature_noise, cae1feature, cae2options);
                            
cae2feature = caefeedForward(optcae2theta, cae1feature, filterDims_Layer2, ...
    numFilters_Layer2, poolDims_Layer2, poolIndex_Layer2, poolMethod, convMethod, 0);
%cae2featureDims = (imageDims - filterDims_Layer1 + 1) ./ poolDims_Layer1;

%% STEP3 softmax layer for classification
%  Randomly initialize the parameters
caesoftmaxtheta = 0.005 * randn(size(cae2feature, 1) * numClasses, 1);

lambda_sfm = 1e-4; 
options_sfm.maxIter = 100;
softmaxModel = softmaxTrain(size(cae2feature, 1), numClasses, lambda_sfm, ... 
                            cae2feature, imagesLabels_train, options_sfm); 
caesoftmaxtheta = softmaxModel.optTheta(:);

%% STEP 4 finetune softmax model 
% Initialize the stack using the parameters learned
stack = cell(layerNum,1);
convLen1 = prod(filterDims_Layer1)*numFilters_Layer1;
stack{1}.w = reshape(optcae1theta(1:convLen1), [filterDims_Layer1 numFilters_Layer1]);
stack{1}.b = optcae1theta(1 + 2 * convLen1 : 2 * convLen1 + numFilters_Layer1);
stack{1}.pd = poolDims_Layer1;
stack{1}.pi = poolIndex_Layer1;
stack{1}.fd = imageDims;
convLen2 = prod(filterDims_Layer2)*numFilters_Layer2;
stack{2}.w = reshape(optcae2theta(1:convLen2), [filterDims_Layer2 numFilters_Layer2]);
stack{2}.b = optcae2theta(1 + 2 * convLen2 : 2 * convLen2 + numFilters_Layer2);
stack{2}.pd = poolDims_Layer2;
stack{2}.pi = poolIndex_Layer2;  
stack{2}.fd = [cae1featureDims(1:2), size(cae1feature, 3)];

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack, poolMethod, convMethod);
stackacetheta = [caesoftmaxtheta ; stackparams ];

%%SGD algorithm for training
stackcaeoptions.epochs = 2;
stackcaeoptions.minibatch = 1;
stackcaeoptions.alpha = 0.01;%1e-1
stackcaeoptions.momentum = .95;

optstackcaetheta = minFuncSGD(@(x, y, z) stackcaeCost(x, y, z, size(cae2feature, 1),...
                                numClasses, netconfig, lambda), ...
                                stackacetheta, images_train, imagesLabels_train, stackcaeoptions);

end