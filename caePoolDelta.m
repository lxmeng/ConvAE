function poolDelta = caePoolDelta(poolDims, convDelta, poolMethod)

if all(poolDims < 2)
    poolDelta = convDelta;
    return;
end

switch poolMethod
    case 'mean'
        numImages = size(convDelta, 4);
        numFilters = size(convDelta, 3);
        convDims = [size(convDelta, 1) size(convDelta, 2)];

        poolDelta = zeros(convDims(1), convDims(2), numFilters, numImages);

        pooledDims = convDims ./ poolDims;
        %PoolImage = zeros(pooledDim1, pooledDim2);
        
        UpsampleMatrix = ones(poolDims);

        for imageNum = 1:numImages
          for featureNum = 1:numFilters
              
              PoolData = squeeze(convDelta(:, :, featureNum, imageNum));
              patches = im2col(PoolData, poolDims, 'distinct');
              meanpatches = mean(patches);
              meanpatches = reshape(meanpatches, pooledDims);
              poolDelta(:, :, featureNum, imageNum) = kron(meanpatches, UpsampleMatrix);
              
          end
        end
        
    case 'max'
        poolDelta = convDelta;
        return;       
end