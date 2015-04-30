function CompareOriRecImages(images, images_rec)

subImRows = size(images_rec, 1);
subImCols = size(images_rec, 2);
numImages = size(images_rec, 3);
PicperRow = 20;
ColInterval = 10;
RowIntervalShort = 5;
RowIntervalLong = 15;
RowNum = ceil(numImages / PicperRow) * 2;
ColNum = PicperRow;
CompShowImage = zeros(subImRows * RowNum + (RowNum / 2 - 1) * RowIntervalLong + RowNum / 2 * RowIntervalShort, ...
    subImCols * ColNum + (ColNum - 1) * ColInterval);
[ColInd, RowInd] = ind2sub([ColNum RowNum/2], 1:numImages);
for Ind = 1 : numImages;
    rposBegin_Ori = 2 * (RowInd(Ind) - 1) * subImRows + (RowInd(Ind) - 1) * RowIntervalLong + ...
       (RowInd(Ind) - 1) * RowIntervalShort + 1;
    cposBegin_Ori = (ColInd(Ind) - 1) * (ColInterval + subImCols) + 1;
    CompShowImage(rposBegin_Ori : rposBegin_Ori + subImRows - 1, ...
        cposBegin_Ori :cposBegin_Ori + subImCols - 1) = images(:,:,Ind);
    
    rposBegin_Rec = (2 * RowInd(Ind) - 1) * subImRows + (RowInd(Ind) - 1) * RowIntervalLong + ...
      RowInd(Ind) * RowIntervalShort  + 1;
    cposBegin_Rec = (ColInd(Ind) - 1) * (ColInterval + subImCols) + 1;
    CompShowImage(rposBegin_Rec : rposBegin_Rec + subImRows - 1,...
        cposBegin_Rec :cposBegin_Rec + subImCols - 1) = images_rec(:,:,Ind);
end
figure(3);
imshow(CompShowImage);

end