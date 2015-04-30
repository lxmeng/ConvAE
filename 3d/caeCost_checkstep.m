function [cost, grad] = caeCost_checkstep(theta, images, data, filterDims, numFilters, poolDims, lambda, poolIndex, poolMethod, convMethod, getCost)
% Calcualte cost and gradient for a CAE with MSE cost objective
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta
if ~exist('getCost','var')    
    getCost = false;
end; 

imageDims = [size(images, 1) size(images, 2), size(images, 3)]; % height/width of image
numImages = size(images, 4); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wvc is filterDim x filterDim x numFilters parameter matrix
% bvc is corresponding bias
[Wc, Wcv, bc, bcv] = caeParamsToStack(theta, filterDims, numFilters);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wcv_grad = zeros(size(Wcv));
bc_grad = zeros(size(bc));
bcv_grad = zeros(size(bcv));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to 
%  reconstruct the original image

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
switch convMethod{1}
    case 'valid'
        convDims = imageDims - filterDims + 1;
    case 'full'
        convDims = imageDims + filterDims - 1; % dimension of convolved output
end
%outputDims = (convDims) ./ poolDims; % dimension of subsampled output 

% convDim x convDim x numFilters x numImages tensor for storing activations

%activations = zeros(convDims(1), convDims(2), numFilters, numImages);
activations = caeConvolve(filterDims, numFilters, images, Wc, bc, convMethod{1});

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations

%activationsPooled = zeros(convDims(1), convDims(2), numFilters, numImages);
activationsPooled = caePool(poolDims, activations, poolIndex, poolMethod); 

%% Reconstruction image
%images_Constr = zeros(size(images));
% images_Constr = caeConvolveBack(filterDims, numFilters, activationsPooled, Wcv, bcv, convMethod{2});

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%

% sparsityParam = 0.01;  
% beta = 3;            % weight of sparsity penalty term  

% edgemask = zeros(imageDims); 
% edgemask(filterDims(1): end - filterDims(1) + 1, ... % + 1
%                      filterDims(2) : end - filterDims(2) + 1, ...
%                      filterDims(3) : end - filterDims(3) + 1) = 1; 
ReconsDiff = (activationsPooled - data) ;%.* repmat(edgemask, [1 1 1 numImages]);             
MSEErr = sum((ReconsDiff(:) .^ 2)) / (numImages * 2) / imageDims(3);
penalty = lambda * (sum(Wc(:).^2)) / 2;%+ sum(Wcv(:).^2)

% activations_sp = squeeze(mean(reshape(activationsPoolBacked, [], numFilters, numImages), 3));
% spPenalty = sparsityParam * log(sparsityParam ./ activations_sp) + (1 - sparsityParam) * log((1 - sparsityParam) ./ (1 - activations_sp));

cost = MSEErr + penalty; %+ beta * sum(spPenalty(:)) / numFilters 
%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

if getCost  
    grad = 0;
    return;
end

%%% YOUR CODE HERE %%%

deltaN = ReconsDiff .* activationsPooled .* (1 - activationsPooled);
% deltaN_ConvVerse = zeros(convDims(1), convDims(2), convDims(3), numFilters, numImages);
% %get Wcv and bcv grad 
% 
% for featureNum = 1:numFilters
%     filter2 = squeeze(Wcv(:, :, :,featureNum));
%     for imageNum = 1:numImages
%         
%         deltaN_Rot = deltaN(:, :, :, imageNum);
%         deltaN_Rot = reshape(deltaN_Rot(end:-1:1), size(deltaN_Rot));
%         switch convMethod{1}
%             case 'valid'
%                 Wcv_grad(:,:,:, featureNum) = Wcv_grad(:,:,:, featureNum) + ...
%                     convn(deltaN_Rot, activationsPooled(:,:,:,featureNum, imageNum), 'valid');
%             case 'full'
%                 Wcv_grad(:,:,:, featureNum) = Wcv_grad(:,:,:, featureNum) + ...
%                     convn(activationsPooled(:,:,:,featureNum, imageNum), deltaN_Rot, 'valid');
%         end
%         
%         deltaN_ConvVerse(:, :, :, featureNum, imageNum) = convn(deltaN(:,:, :,imageNum), filter2, convMethod{1});
%         
%     end
% end
% 
% % delta_sp = beta * (- sparsityParam ./ activations_sp + (1 - sparsityParam) ./ (1 - activations_sp));
% % deltaN_ConvVerse = bsxfun(@plus, deltaN_ConvVerse, reshape(delta_sp, [convDim convDim numFilters]) / numFilters);
% 
% %Wcv_grad = bsxfun(@rdivide, Wcv_grad, sum(Wcv_grad, 3));
% Wcv_grad = Wcv_grad / numImages / imageDims(3) + lambda * Wcv; 
% bcv_grad = sum(deltaN(:)) / numImages / imageDims(3);
% 
% % delta_poolBack = caePool_delta(poolDim, deltaN_ConvVerse);
% % delta_pool = caePoolBack_delta(poolDim, delta_poolBack);
% 
% delta_pool = caePoolDelta(poolDims, deltaN_ConvVerse, poolIndex, poolMethod);
% switch(poolMethod)
%     case 'max'
%         delta_pool = delta_pool .* activationsPooled .* (1 - activationsPooled);
%     case 'mean'
%         delta_pool = delta_pool .* activations .* (1 - activations);
% end

for imageNum = 1:numImages 
  for featureNum = 1:numFilters      
      pooldelta = squeeze(deltaN(:,:,:, featureNum, imageNum));
      pooldelta = reshape(pooldelta(end:-1:1), size(pooldelta));
      im = squeeze(images(:, :, :, imageNum)); 
      switch convMethod{1}
          case 'valid'
              Wc_grad(:, :, :, featureNum) = Wc_grad(:, :, :, featureNum) + convn(im, pooldelta, 'valid');
          case 'full'
              Wc_grad(:, :, :, featureNum) = Wc_grad(:, :, :, featureNum) + convn(pooldelta, im, 'valid');
      end
      bc_grad(featureNum) = bc_grad(featureNum) + sum(pooldelta(:));
  end
end

Wc_grad = Wc_grad / numImages / imageDims(3) + lambda * Wc;
bc_grad = bc_grad / numImages / imageDims(3);

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wcv_grad(:) ; bc_grad(:) ; bcv_grad(:)];

end
