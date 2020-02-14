clc;
clear all;

allImages = imageDatastore('C:\Users\Deep_Learning\Desktop\sign-language-alphabet-recognizer-master\dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[trainingImages, testImages] = splitEachLabel(allImages, 0.8, 'randomized');

alex = alexnet;

layers = alex.Layers

inputSize = alex.Layers(1).InputSize

layers(23) = fullyConnectedLayer(29);
layers(25) = classificationLayer


%Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingImages, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),testImages);
%Until here

opts = trainingOptions('sgdm', 'InitialLearnRate', 1e-4, 'MaxEpochs', 4, 'MiniBatchSize', 100,'ValidationData',augimdsValidation,'ValidationFrequency',3,'ValidationPatience',Inf,'Verbose',false,'ExecutionEnvironment','GPU','Plots','training-progress');

myNet = trainNetwork(augimdsTrain, layers, opts);

predictedLabels = classify(myNet, testImages);
accuracy = mean(predictedLabels == testImages.Labels)

% __________________________________________________________________
% Test recognition with an image
% 
% nnet = alexnet;  % Load the neural net
% 
% picture = imread('C:\Users\sf4e\OneDrive\Kumoh National Institute of Technology\dataset\A\A1.jpg');
% picture = imresize(picture,[227,227]);  % Resize the picture
% label = classify(myNet, picture);        % Classify the picture
% image(picture);     % Show the picture
% title(char(label)); % Show the label
% drawnow;
% 
% end
% ________________________________________________________________

function I = readFunctionTrain(filename)
I = imread(filename);
I = imresize(I, [227 227]);
end