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

% Configuration
imageDims = [28 28];
filterDims = [5 5];    % Filter size for conv layer
numFilters = 10;   % Number of filters for conv layer
poolDims = [4 4];      % Pooling dimension, (should divide imageDim-filterDim+1)
lambda = 1e-4; % Weight decay parameter  
poolMethod = 'max';

% Load MNIST Train
addpath ../softmax/;
images = loadMNISTImages('../../DNN/Dataset/mnist/train-images-idx3-ubyte');
images = reshape(images,imageDims(1), imageDims(2),[]);%(:,1:1000)
imagesLabels = loadMNISTLabels('../../DNN/Dataset/mnist/train-labels-idx1-ubyte');
imagesLabels(imagesLabels == 0) = 10; % Remap 0 to 10
%images = images .* (rand(size(images)) > 0.1);% add noise

%load ..\Codes\DeepLearnToolbox-master\DeepLearnToolbox-master\data\mnist_uint8;
%images = reshape(train_x',imageDim,imageDim,[]);

% Initialize Parameters
theta = caeInitParams(imageDims,filterDims,numFilters,poolDims);     

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
    db_numFilters = 2; 
    db_filterDims = [5 5];  
    db_poolDims = [4 4];
    db_images = images(:,:,1:10);%double(images(:,:,1))/255; 
    
    NomoreInit = 0;    
    for Ind=1:100 
    db_theta = caeInitParams(imageDims, db_filterDims, db_numFilters, db_poolDims);
    [~, grad] = caeCost(db_theta, db_images, db_filterDims, db_numFilters, db_poolDims, lambda, poolMethod);
    

    % Check gradients  
    numGrad = computeNumericalGradient( @(x) caeCost(x,db_images,db_filterDims,...
                                db_numFilters,db_poolDims, lambda, poolMethod, 1), db_theta);
  
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);          
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);     
    % Should be small. In our implementation, these values are usually 
    % less than 1e-9.
    disp(diff);       
    if(diff > 1e-5) 
        NomoreInit = 1;
    end
    end
 
    assert(diff < 1e-9,... 
        'Difference too large. Check your gradient computation again');
end;

TrainIndex = 1 : 1000;

%%SGD algorithm for training
options.epochs = 1;
options.minibatch = 1;
options.alpha = 0.01;%1e-1
options.momentum = .95;
opttheta = minFuncSGD(@(x,y) caeCost(x, y, filterDims,...
                                numFilters, poolDims, lambda, poolMethod), theta, images(:,:, TrainIndex), options);
                            
% addpath ../SAE/minFunc
% options.Method = 'CG';%L-BFGS
% options.display = 'on';
% options.maxIter = 200;% maxIter;
% sae1OptTheta = minFunc( @(p) caeCost(p, images(:,:, TrainIndex), filterDim,...
%                                 numFilters, poolDim, lambda), ...
%                                 theta, options);

%%======================================================================
%% STEP 4: Visualization 

W1 = reshape(opttheta(1:prod(filterDims)*numFilters), prod(filterDims), numFilters);
figure(2);
display_network(W1, 5); 

print -djpeg weights.jpg   % save the visualization to a file

SampleShowIndex = randi([1 size(images, 3)], 100, 1);
images_rec = caefeedForward(opttheta, images(:,:,SampleShowIndex), filterDims, ...
    numFilters, poolDims, poolMethod, 1);
CompareOriRecImages(images(:,:,SampleShowIndex), images_rec);

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

