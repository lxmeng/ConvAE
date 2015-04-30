function convolvedFeatures = caeConvolve(filterDims, numFilters, images, W, b, convMethod)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

numImages = size(images, 4);
imageDims = [size(images, 1), size(images, 2), size(images, 3)];
switch convMethod
    case 'valid'
        convDims = imageDims - filterDims + 1;% valid
    case 'full'
        convDims = imageDims + filterDims - 1;% full
end

convolvedFeatures = zeros(convDims(1), convDims(2), convDims(3), numFilters, numImages); 

% Instructions:
%   Convolve every filter with every image here to produce the 
%   (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
%   matrix convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
%
% Expected running times: 
%   Convolving with 100 images should take less than 30 seconds 
%   Convolving with 5000 images should take around 2 minutes
%   (So to save time when testing, you should convolve with less images, as
%   described earlier)


for imageNum = 1:numImages 
  for filterNum = 1:numFilters

    % convolution of image with feature matrix
    % Obtain the feature (filterDim x filterDim) needed during the convolution

    %%% YOUR CODE HERE %%%
    %feature = zeros(filterDim, filterDim); % You should replace this
    feature = W(:, :, :, filterNum);
 

    % Flip the feature matrix because of the definition of convolution, as explained later
    %feature = rot90(squeeze(feature),2);
    feature = reshape(feature(end:-1:1), size(feature));
      
    % Obtain the image
    im = squeeze(images(:, :, :, imageNum));

    % Convolve "filter" with "im", adding the result to convolvedImage
    % be sure to do a 'valid' convolution

    %%% YOUR CODE HERE %%%
    convolvedImage = convn(im, feature, convMethod);
    % Add the bias unit
    % Then, apply the sigmoid function to get the hidden activation
    
    convolvedImage = bsxfun(@plus, convolvedImage, b(filterNum)); 
    convolvedImage = sigmoid(convolvedImage);
    %%% YOUR CODE HERE %%%
    
    convolvedFeatures(:, :, :, filterNum, imageNum) = convolvedImage;
  end

end
end
  
  % You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end