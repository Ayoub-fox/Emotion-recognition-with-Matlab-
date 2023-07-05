% Script for training the emotion recognition classifier

% Assuming you have a dataset directory containing subfolders for each emotion category
datasetDir = 'C:\Users\Ayoub\Downloads\im_processing_project\train';
categories = {'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'};
imageSize = [64, 64]; % Resizing images to a 64x64 resolution

imds = imageDatastore(datasetDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds.ReadFcn = @(filename) readAndPreprocessImage(filename, imageSize);

% Split the dataset into training and testing sets
[trainSet, testSet] = splitEachLabel(imds, 0.7, 'randomized');

% Assuming you have loaded and preprocessed images in the variables 'trainSet' and 'testSet'

% Determine the feature vector size
sampleImage = readimage(trainSet, 1);
sampleFeatures = extractHOGFeatures(sampleImage);
featureVectorSize = numel(sampleFeatures);

hogFeaturesTrain = zeros(numel(trainSet.Files), featureVectorSize);
hogFeaturesTest = zeros(numel(testSet.Files), featureVectorSize);

for i = 1:numel(trainSet.Files)
    img = readimage(trainSet, i);
    hogFeaturesTrain(i, :) = extractHOGFeatures(img);
end

for i = 1:numel(testSet.Files)
    img = readimage(testSet, i);
    hogFeaturesTest(i, :) = extractHOGFeatures(img);
end

% Train a classifier
classifier = fitcecoc(hogFeaturesTrain, trainSet.Labels);
save('trainedClassifier2.mat', 'classifier');

% Define a helper function to read and preprocess the image
function img = readAndPreprocessImage(filename, imageSize)
    img = imread(filename);
    
    % Check if the image is grayscale
    if size(img, 3) == 1
        img = imresize(img, imageSize);
    else
        img = imresize(rgb2gray(img), imageSize);
    end
end
