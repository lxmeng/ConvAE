function convolvedBackImages = caeConvolveBack(filterDims, numFilters, convolvedImages, W, b)

numImages = size(convolvedImages, 3);
convDims = [size(convolvedImages, 1) size(convolvedImages, 2)];
imageDims = convDims + filterDims - 1;

convolvedBackImages = zeros(imageDims(1), imageDims(2), numImages);

% for imageNum = 1:numImages
%     reconstrImage = zeros(imageDims);
    for filterNum = 1:numFilters
        feature = W(:, :, filterNum);
        feature = rot90(squeeze(feature),2);
        convolvedBackImages = convolvedBackImages + ...
            convn(convolvedImages(:,:,:,filterNum), feature, 'full');
%         im = squeeze(convolvedImages(:, :, imageNum, filterNum));
%         reconstrImage = reconstrImage + conv2(im, feature, 'valid');
    end
    convolvedBackImages = sigmoid(convolvedBackImages + b);
%     convolvedBackImages(:,:,imageNum)= sigmoid(reconstrImage+b);
% end
end
  
  % You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end