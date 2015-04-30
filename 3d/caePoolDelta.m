function poolDelta = caePoolDelta(poolDims, convDelta, poolIndex, poolMethod)

if all(poolDims < 2)
    poolDelta = convDelta;
    return;
end

switch poolMethod
    case 'mean'
        numImages = size(convDelta, 5);
        numFilters = size(convDelta, 4);
        convDims = [size(convDelta, 1) size(convDelta, 2) size(convDelta, 3)];

        poolDelta = zeros(convDims(1), convDims(2), convDims(3), numFilters, numImages);

        %pooledDims = convDims ./ poolDims;
        %PoolImage = zeros(pooledDim1, pooledDim2);
        
        %UpsampleMatrix = ones(poolDims);

        for imageNum = 1:numImages
          for featureNum = 1:numFilters
              PoolData = squeeze(convDelta(:, :, :, featureNum, imageNum));
              patches_mean = mean(PoolData(poolIndex));
              PoolData(poolIndex) = repmat(patches_mean, [size(poolIndex, 1) 1 1]);               
              poolDelta(:, :, :, featureNum, imageNum) = PoolData;
              %{
              old code use for
%               for channel = 1 : pooledDims(3)
%                   PoolData = squeeze(convDelta(:, :, channel, featureNum, imageNum));
%                   patches = im2col(PoolData, poolDims, 'distinct');
%                   meanpatches = mean(patches);
%                   meanpatches = reshape(meanpatches, [pooledDims(1) pooledDims(2)]);
%                   poolDelta(:, :, channel, featureNum, imageNum) = kron(meanpatches, UpsampleMatrix);
%               end
              %}
          end
        end
        
    case 'max'
        poolDelta = convDelta;
        return;       
end