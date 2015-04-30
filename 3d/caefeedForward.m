function Features = caefeedForward(theta, images, filterDims, numFilters, poolDims, poolIndex, poolMethod, convMethod, DoCostr)

if ~exist('DoCostr','var')
    DoCostr = false;
end;

imageDims = [size(images, 1) size(images, 2) size(images, 3)]; % height/width of image
numImages = size(images, 4); % number of images

[Wc, Wcv, bc, bcv] = caeParamsToStack(theta, filterDims, numFilters, DoCostr);

%convDims = imageDims + filterDims - 1; % dimension of convolved output

% convDim x convDim x numFilters x numImages tensor for storing activations
%activations = zeros(convDim,convDim,numFilters,numImages);
activations = caeConvolve(filterDims, numFilters, images, Wc, bc, convMethod{1});

if ~DoCostr
    activationsPooled = caePool(poolDims, activations, poolIndex, poolMethod, 1);
    Features = reshape(activationsPooled, [], numImages);
    return;
else
    activationsPooled = caePool(poolDims, activations, poolIndex, poolMethod);
    Features = caeConvolveBack(filterDims, numFilters, activationsPooled, Wcv, bcv, convMethod{2});
end

end