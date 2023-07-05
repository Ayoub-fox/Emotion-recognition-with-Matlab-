function emotionRecognitionGUI()
    % Create the GUI figure and components
    fig = uifigure('Name', 'Emotion Recognition', 'Position', [100 100 400 300]);
    btnUpload = uibutton(fig, 'Position', [50 200 100 30], 'Text', 'Upload Image', 'ButtonPushedFcn', @uploadImageCallback);
    imgDisplay = uiimage(fig, 'Position', [200 50 150 150]);
    txtEmotion = uitextarea(fig, 'Position', [200 220 150 30], 'Editable', false);

    % Load the trained classifier
    trainedClassifier = load('trainedClassifier2.mat');
    classifier = trainedClassifier.classifier;

    % Define the callback function for the upload button
    function uploadImageCallback(~, ~)
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png'}, 'Select Image');
        if isequal(file, 0)
            return;
        end

        % Read and preprocess the uploaded image
        filename = fullfile(path, file);
        imageSize = [64, 64]; % Set the desired image size
        img = readAndPreprocessImage(filename, imageSize);

        % Extract HOG features
        hogFeatures = extractHOGFeatures(img);

        % Predict the emotion using the trained classifier
        predictedLabel = predict(classifier, hogFeatures);

        % Display the uploaded image and predicted emotion
        imgDisplay.ImageSource = filename;
        txtEmotion.Value = predictedLabel;
    end

    % Helper function to read and preprocess the image
    function img = readAndPreprocessImage(filename, imageSize)
        img = imread(filename);

        % Check if the image is grayscale
        if size(img, 3) == 1
            img = imresize(img, imageSize);
        else
            img = imresize(rgb2gray(img), imageSize);
        end
    end
end
