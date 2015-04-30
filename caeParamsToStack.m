function [Wc, Wcv, bc, bcv] = caeParamsToStack(theta, filterDims, numFilters)
% Converts unrolled parameters for CAE
%                            
% Parameters:
%  theta      -  unrolled parameter vectore
%  imageDim   -  height/width of image
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%
%
% Returns:
%  Wc      -  filterDim x filterDim x numFilters parameter matrix
%  Wvc     -  filterDim x filterDim x numFilters parameter matrix
%  bc      -  bias for convolution layer of size numFilters x 1
%  bvc     -  bias for reconstruction of size 1

%% Reshape theta
indS = 1;  
indE = prod(filterDims) * numFilters;
Wc = reshape(theta(indS:indE), filterDims(1), filterDims(2), numFilters);
indS = indE + 1;
indE = indE + prod(filterDims) * numFilters;
Wcv = reshape(theta(indS:indE), filterDims(1), filterDims(2), numFilters);
indS = indE + 1;
indE = indE + numFilters;
bc = theta(indS : indE);
bcv = theta(indE +1 : end);
% bcv = reshape(bcv, imageDim, imageDim);


end