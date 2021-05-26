%画像取得（フォルダごと）
imds = imageDatastore('images', 'IncludeSubfolders',true,'LabelSource','foldernames');
n_label = countEachLabel(imds);

%割合指定
rateTrainFiles = 0.7;
[imdsTrain, imdsValidation] = splitEachLabel(imds, rateTrainFiles,'randomize');

%argumentation
s_image = [256, 256, 3];
aug_imdsTrain = augmentedImageDatastore(s_image,imdsTrain);
aug_imdsValidation = augmentedImageDatastore(s_image,imdsValidation);


%DNN
layers = [
    imageInputLayer(s_image)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

%training
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',aug_imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(aug_imdsTrain, layers, options);

%% 検証
im_test = imread('test_apple.jpg');
im =imresize(im_test,s_image(1:2)); 
figure(1);
image(im);                   
label = classify(net,im);    d
title(char(label));          
