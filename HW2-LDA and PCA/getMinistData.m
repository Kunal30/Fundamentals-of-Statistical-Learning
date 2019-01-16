function [trX, trY, tsX, tsY] = getMinistData()
% Loads MNIST data into variables 
% trX = [noPixels x noTrSamples] - training data images scaled [0,1]
% trY = [1 x noTrSamples] - train labels in the range [0-9] 
% tsX = [noPixels x noTsSamples] - test data images scaled [0,1]
% tsY = [1 x noTsSamples] - test labels in the range [0-9] 
%
% Created on 9/28/2018 by hkdv1@asu.edu

noTrSamples = 1000;
noTrPerClass = 100;
noTsSamples = 100;
noTsPerClass = 10;
labelsToGet = 0:9; 

trainXFileName = 'mnist/train-images-idx3-ubyte';
trainYFileName = 'mnist/train-labels-idx1-ubyte';
testXFileName = 'mnist/t10k-images-idx3-ubyte';
testYFileName = 'mnist/t10k-labels-idx1-ubyte'; 

% Load train Data and Labels
trImages = loadMNISTImages(trainXFileName);
trLabels = loadMNISTLabels(trainYFileName); 

% Filter data to get required no. of samples 
trX = zeros(size(trImages,1), noTrSamples);
trY = zeros(1,noTrSamples);
for ll = 1:length(labelsToGet)
    idl = find(trLabels == labelsToGet(ll));
    idx = (ll-1)*noTrPerClass+1:ll*noTrPerClass;
    trX(:,idx) = trImages(:,idl(1:noTrPerClass));
    trY(idx) = labelsToGet(ll)*ones(1,noTrPerClass);
end

% Load test Data and Labels
tsImages = loadMNISTImages(testXFileName);
tsLabels = loadMNISTLabels(testYFileName); 

% Filter data to get required no. of samples 
tsX = zeros(size(tsImages,1), noTsSamples);
tsY = zeros(1,noTsSamples);
for ll = 1:length(labelsToGet)
    idl = find(tsLabels == labelsToGet(ll));
    idx = (ll-1)*noTsPerClass+1:ll*noTsPerClass;
    tsX(:,idx) = tsImages(:,idl(1:noTsPerClass));
    tsY(idx) = labelsToGet(ll)*ones(1,noTsPerClass);
end
rng(1)
rd = randperm(noTsSamples);
tsY = tsY(rd);
tsX = tsX(:,rd);
end

function trData = loadMNISTImages(fileName) 
    fp = fopen(fileName, 'rb');
    assert(fp ~= -1, ['Could not open ', fileName, '']);
    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', fileName, '']);
    numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
    numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
    numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
    trData = fread(fp, inf, 'unsigned char');
    trData = reshape(trData, numCols, numRows, numImages);
    trData = permute(trData,[2 1 3]);
    fclose(fp);
    % Reshaping to [noPixels x noSamples]
    trData = reshape(trData, size(trData, 1) * size(trData, 2), size(trData, 3)); %[784 x 60000]
    % Convert to double and rescale to [0,1]
    trData = double(trData) / 255;
end

function labels = loadMNISTLabels(fileName) 
    fp = fopen(fileName, 'rb');
    assert(fp ~= -1, ['Could not open ', fileName, '']);
    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', fileName, '']);
    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
    labels = fread(fp, inf, 'unsigned char');
    assert(size(labels,1) == numLabels, 'Mismatch in label count');
    fclose(fp);
end