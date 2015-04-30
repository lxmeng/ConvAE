function pooledFeatures = caePool(poolDims, convolvedFeatures, poolIndex, poolMethod, getFeature)
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

numImages = size(convolvedFeatures, 5);
numFilters = size(convolvedFeatures, 4);
convDims = [size(convolvedFeatures, 1) size(convolvedFeatures, 2) size(convolvedFeatures, 3)];
pooledDims = convDims ./ poolDims;

if getFeature
    pooledFeatures = zeros(pooledDims(1), pooledDims(2), pooledDims(3), numFilters, numImages);
else
    pooledFeatures = zeros(convDims(1), convDims(2), pooledDims(3), numFilters, numImages);
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
      
      PoolImage = squeeze(convolvedFeatures(:, :, :, featureNum, imageNum));
      switch(poolMethod)
          case 'mean'
              poolpatches_mean = mean(PoolImage(poolIndex));
              if getFeature
                  pooledFeatures(:, :, :, featureNum, imageNum) = reshape(poolpatches_mean, pooledDims);
              else
                  poolpatches = repmat(poolpatches_mean, [size(poolIndex, 1) 1 1]);
                  PoolImage(poolIndex) = poolpatches;
                  pooledFeatures(:, :, :, featureNum, imageNum) = PoolImage;
              end
          case 'max'
              digits(5);
              %PoolImage=PoolImage+rand(size(PoolImage))*1e-12;
              poolpatches = PoolImage(poolIndex);
              [poolpatches_max, max_index] = max(poolpatches);    %floor(poolpatches * 10^5)
              if getFeature 
                  pooledFeatures(:, :, :, featureNum, imageNum) = reshape(poolpatches_max, pooledDims);
              else
                  indexNum = length(max_index(:));
                  [index2 index3] = ind2sub([size(poolIndex,2) size(poolIndex,3)], 1:indexNum);
                  Index = sub2ind(size(poolIndex), max_index(:), index2', index3');                  
                  poolpatches_r = zeros(size(poolIndex));
                  poolpatches_r(Index) = poolpatches_max;%poolpatches(Index); %
                  %poolpatches = poolpatches .* (poolpatches == repmat(poolpatches_max, [size(poolIndex, 1) 1 1]));
                  PoolImage(poolIndex) = poolpatches_r;
                  pooledFeatures(:, :, :, featureNum, imageNum) = PoolImage;
              end
      end 
 %{
%OLD code use for       
      for channel = 1 : size(PoolImage, 3)
          patches = im2col(PoolImage(:, :, channel), poolDims, 'distinct');
          switch(poolMethod)
              case 'mean'
                  poolpatches = mean(patches);
                  poolpatches = reshape(poolpatches, [pooledDims(1) pooledDims(2)]);
                  if getFeature
                      pooledFeatures(:, :, channel, featureNum, imageNum) = poolpatches;
                  else
                      pooledFeatures(:, :, channel, featureNum, imageNum) = kron(poolpatches, UpsampleMatrix);
                  end
              case 'max'
                  %patches = patches + rand(size(patches)) * 1e-12;
                  [m1, m2] = max(patches);
                  if getFeature
                      pooledFeatures(:, :, channel, featureNum, imageNum) = reshape(m1, pooledDims);
                  else
                      patches_R = zeros(size(patches));
                      Index = sub2ind(size(patches), m2, 1 : size(patches,2));
                      patches_R(Index) = m1; 
                      PoolImage(:,:,channel) = col2im(patches_R, poolDims, convDims, 'distinct');   
                      pooledFeatures(:, :, channel, featureNum, imageNum) = PoolImage(:,:,channel);
                  end
          end 
      end
%}
  end
end

end

