function theta = caeInitParams(imageDims,filterDims,numFilters,poolDims)
% Initialize parameters for a single layer convolutional neural
% network followed by a softmax layer.
%                            
% Parameters:
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%
%
% Returns:
%  theta      -  unrolled parameter vector with initialized weights

%% Initialize parameters randomly based on layer sizes.
assert(all(filterDims < imageDims),'filterDims must be less that imageDims');

r  =  numFilters * prod(filterDims) + prod(filterDims);

Wc = (rand(filterDims(1), filterDims(2), numFilters) - 0.5) * 2 * sqrt(6 / r);%convolution kernels

outDims = imageDims + filterDims - 1; % dimension of convolved image

% assume outDim is multiple of poolDim
assert(all(mod(outDims,poolDims)==0),...
       'poolDim must divide imageDim - filterDim + 1');
   
Wcv = (rand(filterDims(1), filterDims(2), numFilters)- 0.5) * 2 * sqrt(6 / r);

bc = zeros(numFilters, 1);
bcv = 0;

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [Wc(:) ; Wcv(:) ; bc(:) ; bcv(:)];

end

