function poolIndex = getpoolIndex(imageDims, filterDims, poolDims, convMethod)
switch convMethod 
    case 'valid'
        convDims = imageDims - filterDims + 1;
    case 'full'
        convDims = imageDims + filterDims - 1;
end
OriIndex = reshape(1:prod(convDims), convDims);
poolIndex = zeros(prod(poolDims), prod(convDims(1:2)) / prod(poolDims), convDims(3));
for channel = 1 : convDims(3)
    poolIndex(:, :, channel) = im2col(OriIndex(:,:,channel), poolDims, 'distinct');
end
end