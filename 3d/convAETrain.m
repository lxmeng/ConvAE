function convAETrain
%% CS294A/CS294W Programming Assignment Starter Code

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  programming assignment. You will need to complete the code in sampleIMAGES.m,
%  convAutoencoderCost.m and computeNumericalGradient.m. 
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.
clear;
clc;
close all;
warning off;
rand('state', 0);

% Configuration
imageDims = [64 64 3];%64 64 3
filterDims = [5 5 3];    % Filter size for conv layer
numFilters = 50;   % Number of filters for conv layer
poolDims = [2 2 1];      % Pooling dimension, (should divide imageDim-filterDim+1)
lambda = 1e-4; % Weight decay parameter  
poolMethod = 'mean';
convMethod = { 'full' 'valid'};
poolIndex = getpoolIndex(imageDims, filterDims, poolDims, convMethod{1});
noise = 0.3;
 
% Load MNIST Train
% addpath ../../../DNN/softmax/;
% images = loadMNISTImages('../../../DNN/Dataset/mnist/train-images-idx3-ubyte');
% images = reshape(images,imageDims(1), imageDims(2), imageDims(3),[]);%(:,1:1000)
% imagesLabels = loadMNISTLabels('../../../DNN/Dataset/mnist/train-labels-idx1-ubyte');
% imagesLabels(imagesLabels == 0) = 10; % Remap 0 to 10 
% TrainIndex = 1 : 1000;
% images_train = images(:,:,:, TrainIndex);
% images_train_noise = images_train .* (rand(size(images_train)) > noise);

%load STL-100patch
% addpath ../../../DNN/linearCoder/
% load ../../../DNN/Dataset/stl10_patches_100k/stlSampledPatches.mat
% epsilon = 0.1;
% displayColorNetwork(patches(:, 1:100));
% % Subtract mean patch (hence zeroing the mean of the patches)
% meanPatch = mean(patches, 2);  
% patches = bsxfun(@minus, patches, meanPatch);
% % Apply ZCA whitening
% numPatches = size(patches, 2);
% sigma = patches * patches' / numPatches;
% [u, s, v] = svd(sigma);
% ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
% patches = ZCAWhite * patches;
% displayColorNetwork(patches(:, 1:100));
% images = reshape(patches, imageDims(1), imageDims(2), imageDims(3), []);

%load STL
load('F:\Lingxun.Meng\DNN\Dataset\stlSubset\stlTrainSubset.mat');
images = trainImages;
TrainIndex = 1 : 1000;
images_train = images(:,:,:, TrainIndex);
images_train_noise = images_train .* ((rand(size(images_train))) > noise);

%load ..\Codes\DeepLearnToolbox-master\DeepLearnToolbox-master\data\mnist_uint8;
%images = reshape(train_x',imageDim,imageDim,[]);

% Initialize Parameters
theta = caeInitParams(imageDims,filterDims,numFilters,poolDims, convMethod{1});     

%%======================================================================
%% STEP 1: Implement convAE Objective
%  Implement the function caeCost.m.
%%======================================================================

%%======================================================================
%% STEP 2: Gradient Check
%  Use the file computeNumericalGradient.m to check the gradient
%  calculation for your cnnCost.m function.  You may need to add the
%  appropriate path or copy the file to this directory.

DEBUG = false;  % set this to true to check gradient 
if DEBUG   
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set 
    db_numFilters = 1;  
    db_filterDims = [5 5 3];  
    db_poolDims = [2 2 1];
    db_poolIndex = getpoolIndex(imageDims, db_filterDims, db_poolDims, convMethod{1});
    db_images = images(:,:,:, 1);%double(images(:,:,1))/255;  
    for i = 1 :100   
    db_theta = caeInitParams(imageDims, db_filterDims, db_numFilters, db_poolDims, convMethod{1});
    db_images_noise = db_images .* (rand(size(db_images)) > noise);
    
%     [~, grad] = caeCost(db_theta, db_images_noise, db_images, db_filterDims, db_numFilters, db_poolDims, lambda, db_poolIndex, poolMethod, convMethod);
%      
%  
%     % Check gradients 
%     numGrad = computeNumericalGradient( @(x) caeCost(x,db_images_noise, db_images,db_filterDims,...
%                                 db_numFilters,db_poolDims, lambda, db_poolIndex, poolMethod, convMethod, 1), db_theta);
                            
    load a;
    [~, grad] = caeCost_checkstep(db_theta, db_images_noise, a, db_filterDims, db_numFilters, db_poolDims, lambda, db_poolIndex, poolMethod, convMethod);
     
 
    % Check gradients 
    numGrad = computeNumericalGradient( @(x) caeCost_checkstep(x,db_images_noise, a, db_filterDims,...
                                db_numFilters,db_poolDims, lambda, db_poolIndex, poolMethod, convMethod, 1), db_theta); 
  
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);              
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);     
    % Should be small. In our implementation, these values are usually 
    % less than 1e-9.
    disp(diff);          
    if diff > 1e-4        
        tt = 1;
    end
    end
 
    assert(diff < 1e-9,...   
        'Difference too large. Check your gradient computation again');
end;


%%SGD algorithm for training
options.epochs = 2;
options.minibatch = 1;
options.alpha = 0.01;%1e-1
options.momentum = .95;

opttheta = minFuncSGD(@(x, y, z) caeCost(x, y, z, filterDims,...
                                numFilters, poolDims, lambda, poolIndex, poolMethod, convMethod), theta, images_train_noise, images_train, options);
                            
% addpath ../SAE/minFunc
% options.Method = 'CG';%L-BFGS
% options.display = 'on';
% options.maxIter = 200;% maxIter;
% sae1OptTheta = minFunc( @(p) caeCost(p, images(:,:, TrainIndex), filterDim,...
%                                 numFilters, poolDim, lambda), ...
%                                 theta, options);

%%======================================================================
%% STEP 4: Visualization 
addpath ../../../DNN/linearCoder
W1 = reshape(opttheta(1:prod(filterDims)*numFilters), prod(filterDims), numFilters);
figure(2);
if size(images, 3) > 1
    displayColorNetwork(W1(:, 1:numFilters));
else
    display_network(W1, 5); 
end

print -djpeg weights_o_r_mean.jpg   % save the visualization to a file 

SampleShowIndex = randi([1 size(images, 4)], 100, 1); 
images_rec = caefeedForward(opttheta, images(:, :, :,SampleShowIndex), filterDims, ...
    numFilters, poolDims, poolIndex, poolMethod, convMethod, 1);
CompareOriRecImages(images(:, :, :,SampleShowIndex), images_rec); 
print -djpeg comp_o_r_mean.jpg
%%======================================================================
%% STEP 5: TEST 
% addpath ../softmax 
% 
% testData = loadMNISTImages('../Dataset/mnist/t10k-images-idx3-ubyte');
% testData = reshape(testData,imageDim,imageDim,[]);
% testLabels = loadMNISTLabels('../Dataset/mnist/t10k-labels-idx1-ubyte');
% testLabels(testLabels == 0) = 10; % Remap 0 to 10
% numClasses = 10;
% 
% caeFeatures_Train = caefeedForward(opttheta, images(:, :, TrainIndex), filterDim, numFilters, poolDim * 4);
% 
% lambda_sfm = 1e-4;
% options_sfm.maxIter = 100;
% softmaxModel = softmaxTrain(size(caeFeatures_Train, 1), numClasses, lambda_sfm, ...
%                             caeFeatures_Train, imagesLabels(TrainIndex), options_sfm);
% caeSoftmaxOptTheta = softmaxModel.optTheta(:);
% 
% caeFeatures_Test = caefeedForward(opttheta, testData, filterDim, numFilters, poolDim * 4);
% [pred] = softmaxPredict(softmaxModel, caeFeatures_Test);
% 
% acc = mean(testLabels(:) == pred(:));
% fprintf('Accuracy: %0.3f%%\n', acc * 100);
end
