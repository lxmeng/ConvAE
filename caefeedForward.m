function Features = caefeedForward(theta, images, filterDims, numFilters, poolDims, poolMethod, DoCostr)

if ~exist('DoCostr','var')
    DoCostr = false;
end;

imageDims = [size(images, 1) size(images, 2)]; % height/width of image
numImages = size(images, 3); % number of images

[Wc, Wcv, bc, bcv] = caeParamsToStack(theta, filterDims, numFilters);

%convDims = imageDims + filterDims - 1; % dimension of convolved output

% convDim x convDim x numFilters x numImages tensor for storing activations
%activations = zeros(convDim,convDim,numFilters,numImages);
activations = caeConvolve(filterDims, numFilters, images, Wc, bc);
activationsPooled = caePool(poolDims, activations, poolMethod);
if ~DoCostr
    Features = reshape(activationsPooled, [], numImages);
    return;
else
    Features = caeConvolveBack(filterDims, numFilters, activationsPooled, Wcv, bcv);
end

end