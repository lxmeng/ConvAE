function pooledFeatures = caePool(poolDims, convolvedFeatures, poolMethod)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
% 
if all(poolDims < 2)
    pooledFeatures = convolvedFeatures;
    return;
end

if ~exist('getFeature','var')
    getFeature = false;
end

numImages = size(convolvedFeatures, 3);
numFilters = size(convolvedFeatures, 4);
convDims = [size(convolvedFeatures, 1) size(convolvedFeatures, 2)];
pooledDims = convDims ./ poolDims;

if getFeature
    pooledFeatures = zeros(pooledDims(1), pooledDims(2), numImages, numFilters);
else
    pooledFeatures = zeros(convDims(1), convDims(2), numImages, numFilters);
end

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   mean-pooling or max-pooling

UpsampleMatrix = ones(poolDims);

for imageNum = 1:numImages
  for featureNum = 1:numFilters
      
      PoolImage = squeeze(convolvedFeatures(:, :, imageNum, featureNum));
      
      patches = im2col(PoolImage, poolDims, 'distinct');
      switch(poolMethod)
          case 'mean'
              poolpatches = mean(patches);
              poolpatches = reshape(poolpatches, pooledDims);
              if getFeature
                  pooledFeatures(:, :, imageNum, featureNum) = poolpatches;
              else
                  pooledFeatures(:, :, imageNum, featureNum) = kron(poolpatches, UpsampleMatrix);
              end
          case 'max'
              %patches = patches + rand(size(patches)) * 1e-12;
              [m1, m2] = max(patches);
              if getFeature
                  pooledFeatures(:, :, imageNum, featureNum) = reshape(m1, pooledDims);
              else
                  patches_R = zeros(size(patches));
                  Index = sub2ind(size(patches), m2, 1 : size(patches,2));
                  patches_R(Index) = m1;
                  PoolImage = col2im(patches_R, poolDims, convDims, 'distinct');   
                  pooledFeatures(:, :, imageNum, featureNum) = PoolImage;
              end
      end 
  end
end

end

