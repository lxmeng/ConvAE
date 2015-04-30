function convolvedBackImages = caeConvolveBack(filterDims, numFilters, convolvedImages, W, b, convMethod)

numImages = size(convolvedImages,5);
convDims = [size(convolvedImages, 1) size(convolvedImages, 2) size(convolvedImages, 3)];

switch convMethod
    case 'valid'
        imageDims = convDims - filterDims + 1;
    case 'full'
        imageDims = convDims + filterDims - 1;
end

convolvedBackImages = zeros(imageDims(1), imageDims(2), imageDims(3), numImages);

for imageNum = 1:numImages
    reconstrImage = zeros(imageDims);
    for filterNum = 1:numFilters
        feature = W(:, :, :, filterNum);
        feature = reshape(feature(end:-1:1), size(feature));
        %feature = rot90(squeeze(feature),2);
        im = squeeze(convolvedImages(:, :, :, filterNum, imageNum));
        reconstrImage = reconstrImage + convn(im, feature, convMethod);
    end
    convolvedBackImages(:, :, :,imageNum)= sigmoid(reconstrImage+b);
end
end
  
  % You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end