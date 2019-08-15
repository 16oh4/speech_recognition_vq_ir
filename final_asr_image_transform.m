close all
clear sound
format short
%profile on


%Bruno E. Gracia Villalobos
%Independent Study
%Speech Recognition Using Image Recognition and Vector Quantization
%Professor Grigoryan

%~~~~~~~~~~~~~~~CONFIG PARAMETERS~~~~~~~~~~~~~~~~~~~%
%   ************** Please rename speaker training voices as s1,s2,s3,s4......
trainingSamplesFolder = "database/train/";
testingSamplesFolder = "database/test/";
numSpeakers = 8;        %how many speakers are in each folder
% MFCC PARAMETERS
numMelFilters = 40;     %default is 40
upperLim = 12500;       %this is best dependent on sampling freq of signals (in Hz)
lowerLim = 40;             %usally this low (in Hz)
% SPEECH FRAMING
N = 256;                    %frame length
M = 100;                    %num samples before overlap
overlap = N-M;          %frame overlap
% VECTOR QUANTIZATION
VQDim = [30,4];             %which MFCC numbers to use for VQ section
VQThreshold = 0.005;    %MFCC's under this value will be set to 0
VQOffset = 0.00;            %quantity to add to MFCC's under VQThreshold
sizeCodeBook = 4;           %Use POWERS OF 2. (2, 4, 8...)
% IMAGE RECOGNITION
chunks = 4;                 %How many image "frames" to use (2,4)
numContours = 25;     %how many iterations for active contour (use >25)
%~~~~~~~~~~~~~~~CONFIG PARAMETERS~~~~~~~~~~~~~~~~~~~%

%Check dimensions exist within range of 1 to numMelFilters
if( (VQDim(1) > numMelFilters) || (VQDim(1) < 1) )
    VQDim(1) = floor(numMelFilters/2); %default
elseif( (VQDim(2) > numMelFilters) || (VQDim(2) < 1) )
    VQDim(2) = numMelFilters; %default;
end


fprintf('Calculating MFCCs\n');
tic


%This code block below used to record your own voice. Add 1 to the number
%speakers above if used. 
%{ 
%TRAIN
r = audiorecorder(12500, 16, 1);

disp('Start speaking.')
recordblocking(r, 3);
disp('End of Recording.');

play(r);

samples = getaudiodata(r);

audiowrite('database/train/s9.wav', samples, r.SampleRate);

%TEST

pause(5);

r = audiorecorder(12500, 16, 1);

disp('Start speaking.')
recordblocking(r, 3);
disp('End of Recording.');

play(r);

samples = getaudiodata(r);

audiowrite('database/test/s9.wav', samples, r.SampleRate);

%}

%cell array to store speaker data
[trainArr, trainFS] = getArr(numSpeakers, trainingSamplesFolder); 
[testArr, testFS] = getArr(numSpeakers, testingSamplesFolder);


%~~~~~~~~~~~~~~~STEP 1: Get Speech Frames~~~~~~~~~~~~~~~~~~~%

trainFrameArr = getFrames(trainArr, numSpeakers, N, M);
testFrameArr = getFrames(testArr, numSpeakers, N, M);

%~~~~~~~~~~~~~~~STEP 2: Get MFCC's ~~~~~~~~~~~~~~~~~~~~~~~%

trainMFCC = getMFCC(trainFrameArr, numSpeakers, N, numMelFilters, trainFS);
testMFCC = getMFCC(testFrameArr, numSpeakers, N, numMelFilters, trainFS);

toc
fprintf('\n');

%{
%Plot the MFCC's
plotMFCC(trainMFCC, numMelFilters, numSpeakers);
plotMFCC(testMFCC, numMelFilters, numSpeakers);
%}


%~~~~~~~~~~~~~~~STEP 3: Vector Quantization~~~~~~~~~~~~~~~~~~%

%Offset MFCC's to prevent clustering near 0.
trainMFCCThresh = trainMFCC;
testMFCCThresh = testMFCC;

%{
%If MFCC's are less than threshold set to 0
for j=1:1:numSpeakers
    trainMFCCThresh{j} ( abs(trainMFCC{j}) < VQThreshold ) = 0;

    testMFCCThresh{j} ( abs(testMFCC{j}) < VQThreshold ) = 0;
end
%}

%Plot the training and test MFCC's
plotMFCC(trainMFCCThresh, numMelFilters, numSpeakers);
plotMFCC(testMFCCThresh, numMelFilters, numSpeakers);

fprintf('Creating Codebooks for Vector Quantization\n');
tic

%Select two dimensions from the 40 MFCC's for analyzing in the VQuantizer
trainMFCCVec = getMFCCVectors(trainMFCCThresh, numSpeakers, VQDim);
testMFCCVec = getMFCCVectors(testMFCCThresh, numSpeakers, VQDim);

%Generate the codebook for the training data
trainVQCodeBook = getCodeBook(trainMFCCVec, numSpeakers, sizeCodeBook);
testVQCodeBook = getCodeBook(testMFCCVec, numSpeakers, sizeCodeBook);

%Get the centroids from the codebook
trainCentroidCB = getSmallCodeBook(trainVQCodeBook, numSpeakers, sizeCodeBook);
testCentroidCB = getSmallCodeBook(testVQCodeBook, numSpeakers, sizeCodeBook);

toc
fprintf('\n');

%Plot training data
plotMFCCVectors( trainMFCCVec, numSpeakers, VQDim);
plotCentroids(trainCentroidCB, numSpeakers, sizeCodeBook);

%Plot testing data
plotMFCCVectors( testMFCCVec, numSpeakers, VQDim);
plotCentroids(testCentroidCB, numSpeakers, sizeCodeBook);

fprintf('Matching Train Speakers to Test Speakers\n');
tic

%Find the matched speakers
matchedSpeakers = getMatchedSpeakers(trainCentroidCB, testCentroidCB, ...
    numSpeakers, sizeCodeBook);

toc
fprintf('\nMatches found.\n');

trainStr = strcat('Test:__', num2str(1:1:8));
disp(trainStr);
testStr = strcat('Match:', num2str(matchedSpeakers));
disp(testStr);

%fprintf('The vector below shows the testing speaker(index) closest training match\n');
%disp(matchedSpeakers);
%fprintf('\nFor example, index 1 of the vector is Speaker 1\nand it shows the calculated recognized speaker\n');

%Calculate match for each speaker
ctr=0.0;
for j=1:1:numSpeakers
    if(matchedSpeakers(j) == j)
        ctr = ctr+1;
    end
end

fprintf('Recognition rate: %f %% \n', 100*ctr/numSpeakers);

%}

%~~~~~~~~~~~~~~ASR With Image Recognition~~~~~~~~~~~~~~~~~~~%

%Create cell array to store Sorensen-Dice coefficients for each speaker
SDCBank = cell(1, numSpeakers);
matches = zeros(1,numSpeakers);

index = [1:floor(numMelFilters/chunks):numMelFilters numMelFilters];

%{
figure;
tb = getMFCCimg(trainMFCC{1}, 16);
imshow(tb);

figure;
tg = getMFCCimg(testMFCC{1}, 16);
imshow(tg);
%}
fprintf('\nIR Model: Matching speakers\n');
tic

%This loop compares each test speaker to all training speakers
for i=1:1:numSpeakers %test speakers
    
    SDCBank{2,i} = zeros(1, numSpeakers);
    for k=1:1:numSpeakers %training speakers
        
        %Prepare array to store SDC's of each MFCC frame
        SDCBank{1,i}{1,k} = zeros(1,length(index)-1);
        
        %******* 
        
        size = length(index)-1;
        for j=1:1:size %MFCC chunks
            %fprintf('Testing: %i Training: %i Range: [%i, %i]\n', i, k, index(j), index(j+1));
            
            %Convert MFCC for each speaker into a 16 bit image
            test = getMFCCimg(testMFCC{i}(:, index(j):index(j+1)), 16);
            train = getMFCCimg(trainMFCC{k}(:, index(j):index(j+1)), 16);
            %{
            figure
            subplot(1,2,1);
            imshow(train);
            
            subplot(1,2,2);
            imshow(test);
            %}
            %If the image is blank, skip SDC calculation (sometimes
            %the frames are all blank due to the nature of speaker's MFCC's
            if( ( all(test == 0,'all') || all(train == 0, 'all')) ) 
                %disp('Blank img\n');
                SDC = 0;             
            else
                SDC = getSDC(train, test, numContours); %Get the SDC with Train and Test images
                SDCBank{1,i}{1,k}(j) = SDC; %Store the SDC in the ith'speakers cellarray
            end
        end
        
        %Whenever SDC=0 above, it adds unsignificant data to calculate mean
        %therefore we select only the positive numbers.
        SDCBank{2,i}(k)= mean(nonzeros(SDCBank{1,i}{1,k}).');
        
    end
    %The speaker match will be the training speaker with the highest mean
    matches(i) = find(SDCBank{2,i} == max(SDCBank{2,i}));
end


toc
fprintf('\nMatches found.\n');

trainStr = strcat('Test:__', num2str(1:1:8));
disp(trainStr);
testStr = strcat('Match:', num2str(matches));
disp(testStr);

%Calculate match for each speaker
ctr=0.0;
for j=1:1:numSpeakers
    if(matches(j) == j)
        ctr = ctr+1;
    end
end

fprintf('Recognition rate: %f %% \n', 100*ctr/numSpeakers);
%profile off
%profview

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%~~~~~~~~~~~~~~Image Recognition Functions~~~~~~~~~~~~~~~~~~~%

%Returns the Sorensen Dice Coefficient of two images cropped for accuracy


function SDC = getSDC(trainIMGMFCC, testIMGMFCC, numContours)
%Get values that are not zero
[trainx,trainy] = find(trainIMGMFCC > 0);
[testx,testy] = find(testIMGMFCC > 0);

%Get domain and range from training data
xone = min(trainx);
xtwo = max(trainx);
yone = min(trainy);
ytwo = max(trainy);

%Crop image for actual data, removing the black spaces
TRAINIMG = trainIMGMFCC(xone:xtwo, yone:ytwo);

%Get domain and range from testing data
xone = min(testx);
xtwo = max(testx);
yone= min(testy);
ytwo = max(testy);

if((ytwo-yone <2) & (yone>4))
    yone = yone-4;
end
%Crop image for actual data, removing the black spaces
TESTIMG = testIMGMFCC(xone:xtwo, yone:ytwo);

%{
figure;
subplot(1,2,1);
imshow(TRAINIMG);
title('Cropped Train IMG');

subplot(1,2,2);
imshow(TESTIMG);
title('Cropped Test IMG');
%}

%Get the maximum number of available lines from cropped imgs
rowTR = size(TRAINIMG, 1);
rowTE = size(TESTIMG, 1);

%Pick the image with the most lines to serve as reference for new image
needR = max([rowTR rowTE]);

%Get the maximum number of available MFCC columns from cropped imgs
colTR = size(TRAINIMG, 2);
colTE = size(TESTIMG, 2);

%Select the minimum number of MFCC's to display
needC = min([colTR colTE]);

%If the training image is smaller in row size than the testing image, fill with
%empty data
if(needR > size(TRAINIMG,1))
    TRAINIMG = [TRAINIMG(1:end, 1:needC); zeros(needR-size(TRAINIMG,1), needC)];
%If the testing image is smaller in row size than training image, fill with
%empty data
elseif(needR > size(TESTIMG,1))
    TESTIMG = [TESTIMG(1:end, 1:needC); zeros(needR-size(TESTIMG,1), needC)];
end

%Create new cropped images with the most available data as possible
TRAINIMG = TRAINIMG(1:needR, 1:needC);
TESTIMG = TESTIMG(1:needR, 1:needC);

%Plot trimmed images
%{
figure;
subplot(1,5,1);
imshow(TRAINIMG);
title('Trimmed Train');

subplot(1,5,2);
imshow(TESTIMG);
title('Trimmed Test');
%}
%Sorensen Dice Coefficient
newTestIMG = TESTIMG;
newTrainIMG = TRAINIMG;

%Create contour mask for training image
maskTrain = false(size(newTrainIMG)); 
maskTrain(1:end, 1:end) = true;

%Create contour image for training data using the mask
BWTrain = activecontour(newTrainIMG, maskTrain, numContours);

%{
subplot(1,5,3);
imshow(BWTrain);
title('Train contour');
%}

%Create contour mask for testing image
maskTest = false(size(newTestIMG)); %background is false = black
maskTest(1:end, 1:end) = true; %foreground is white = true

%Create contour image for testing data using the mask
BWTest = activecontour(newTestIMG, maskTest, numContours);

%{
subplot(1,5,4);
imshow(BWTest);
title('Test contour');
%}

%Calculate sorensen dice coeff
SDC = dice(BWTest, BWTrain );
%fprintf('Dice coeff: %f\n', SDC);

%{
%display both contours overlapped
subplot(1,5,5);
imshowpair(BWTest,BWTrain);
title(strcat('SDC: ', num2str(SDC)));
%}

end

%~~~~~~~~~~~~~~Vector Quantization Functions~~~~~~~~~~~~~~~~~~~%

%Creates an image with an MFCC array with specified bits 8/16
function MFCCimg = getMFCCimg(array, bits)

trainIMG = array;

row = size(trainIMG, 1); %number of frames
col = size(trainIMG, 2); %number of MFCC's 

MFCCVector = reshape(trainIMG.', 1, []); %transform frames of mfccs to 1 row vector
normalMFCCVector = MFCCVector; %for storing normalized data from 0 to 256

%for using normalizing equation
minval = min(MFCCVector); 
maxval = max(MFCCVector);

for i=1:1:length(MFCCVector)
    %puts in range 0 to 2^bits
    normalMFCCVector(i) = floor(2^bits * (MFCCVector(i)-minval) ...
        / (maxval - minval) ); 
end

normalMFCCArr = reshape(normalMFCCVector, col, row).'; %transform back to frames x MFCC's array

zeroVal = min(mode(normalMFCCArr)); %this can approximate the normalized value for 0
maxVal = max(max(normalMFCCArr));

%quantize to integer from double using round nearest integer
%remove unwanted data
if(bits == 8)
    MFCCimg = uint8(normalMFCCArr);
elseif(bits ==16)
    MFCCimg = uint16(normalMFCCArr);
    %MFCCimg( MFCCimg < (zeroVal+5000 )) = 0; %7/8 recognition
    MFCCimg( MFCCimg < (zeroVal+5000 )) = 0; %7/8 recognition
elseif(bits==32)
    MFCCimg = uint32(normalMFCCArr);
    MFCCimg( MFCCimg < (zeroVal+11^9-1*10^6) ) = 0; %for using 32 bits
end

end

%Find possible speaker matches given Codebook
function matchedSpeakers = getMatchedSpeakers(trainCentroidCB, ... 
    testCentroidCB, numSpeakers, sizeCodeBook)
%each ith element here will contain the number of the closest speaker
matchedSpeakers = zeros(1, numSpeakers);

row = log2(sizeCodeBook);
%Compute ASR, Vector Quantize the testing data to training centroids

for i=1:1:numSpeakers %traversing the testing data
    
    %this vector will store the sum of the sums of distances currentDistSum
    closestSpeakerDistSum = zeros(1, numSpeakers);
    
    for j=1:1:numSpeakers %traversing the training data
        
        currentDistSum = zeros(1,sizeCodeBook);
        for k=1:1:sizeCodeBook %checking each centroid
            currentTest = testCentroidCB{i}{row, k};
            
            %keep track of the distances between the currentTest centroid
            %and ALL training Centroids for the current speaker
            currentDist = zeros(1,sizeCodeBook); 
            
            %create a vector of the sum of currentDist

            for l=1:1:sizeCodeBook
                currentTrain = trainCentroidCB{j}{row,l};
                %(ref - test)^2
                %currentDist(l) = sqrt(( currentTrain(1) - currentTest(1) )^2 + ...
                        %( currentTrain(2) - currentTest(2) )^2);
                    
                currentDist(l) = ( currentTrain(1) - currentTest(1) )^2 + ...
                        ( currentTrain(2) - currentTest(2) )^2;
                    
                currentDist(l) = sqrt(currentDist(l));
            end
            
            %this vector represents the sum of distances for each TEST centroid
            %compared against each TRAIN centroid
            %currentDistSum(k) = min(currentDist);
            currentDistSum(k) = min(currentDist);
            
            %closest = find(dist == min(dist)); %get index of closest codeword
            %minDist = min(dist);
        end
        
        %Store the total distances computed by currentDistSum for the ith
        %TEST speaker compared against the jth TRAIN speaker
        closestSpeakerDistSum(j) = min(currentDistSum);
        
    end
    
    %select the closest speaker with the index of the minimum VQ distortion
    matchedSpeakers(i) = find(closestSpeakerDistSum == min(closestSpeakerDistSum));
end

end

%Create codebook vectors from the MFCC array
function MFCCvectors = getMFCCVectors(MFCCarray, numSpeakers, VQDim)
%%%% VECTOR QUANTIZATION  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%gather information about the MFCC database

%contains the mean value of each frame per speaker
MFCCavg = cell(1, numSpeakers); 

%contains the 4th and 16th mfcc of each frame per speaker
MFCCvectors = cell(1,numSpeakers); 

%%%~~~~~~~~~~~~~ FOR PLOTTING DATA ~~~~~~~~~~~~~~

%contains the min and max values of MFCC for each speaker
MFCCranges = cell(1, numSpeakers); 

%temp values for figuring out range of x and y
minx = 0;
maxx = 0;
miny = 0;
maxy = 0;

%contains x and y axes for plotting the vector MFCC's
MFCCaxes = cell(1,numSpeakers);

%%%~~~~~~~~~~~~~ FOR PLOTTING DATA ~~~~~~~~~~~~~~

for i=1:1:numSpeakers
    %how many frames in the current speaker
    framesSpeaker = size(MFCCarray{i}, 1);
    
    for j=1:1:framesSpeaker
        %MFCCavg{i}(j) = mean(MFCCarray{i}(j,:));
        
        %create vectors using 4th and 16th dimensions of 40 MFCC's
        MFCCvectors{i}(j,:) = [MFCCarray{i}(j,VQDim(1)) MFCCarray{i}(j,VQDim(2))];
        %MFCCvectors{i}(j,:) = MFCCvectors{i}(j,:);
    
    end
    %calculate minimum values for 2d space
    
    minx = min(MFCCvectors{i}(:, 1)); %get the minimum value of the X column
    maxx = max(MFCCvectors{i}(:, 1)); %get the max value of the X column
    miny = min(MFCCvectors{i}(:, 2));  %get min value of the Y column
    maxy = min(MFCCvectors{i}(:, 2));  %get min value of the Y column
    
    %store the min and max for x and y in a cellarray
    MFCCranges{i} = [minx maxx; miny maxy];
    
   %{ 
        for the first speaker MFCCranges{1} this is the output:
        [min x, max x]
        [min y, max y]
    
    %}
    
    %generate x axis for first speaker
    MFCCaxes{i}{1} = linspace(minx, maxx, framesSpeaker);
    MFCCaxes{i}{2} = linspace(miny, maxy, framesSpeaker);
end


end

%Plot the centroids of the codebook
function plotCentroids(centroidCB, numSpeakers, sizeCodeBook)
    rows = log2(sizeCodeBook);
    powers = 1:1:rows;
    %col = 2.^powers;
    
    %figure;
    %titleStr = strcat('Centroids Plot For All Codebooks With Size: ', num2str(sizeCodeBook));
    %title(titleStr);
    colors = ['b', 'k', 'r', 'g', 'y', 'c', 'm', [0 .5 .75]];
    
    printed = 0; %for displaying only speaker legend once
    
    grid on
    %hold on
    for i=1:1:numSpeakers
        index = mod(i, length(colors)) + 1;
            for k=1:1:sizeCodeBook
                
                legendTitle = strcat('Centroid ', num2str(i));
                
                if(printed==0)
                    scatter(centroidCB{i}{rows,k}(:,1), centroidCB{i}{rows,k}(:,2), 50, ...
                         'd', colors(index), 'DisplayName', legendTitle);
                    
                    printed = 1;
             
                else
                    scatter(centroidCB{i}{rows,k}(:,1), centroidCB{i}{rows,k}(:,2), 50,...
                          'd', colors(index));
                    
                end
            end
        printed = 0; %reset boolean so that the legend for new speaker shows
    end
    legend
    hold off

end

%Remove the intermediate results to get the codebook vectors specified by
%sizeCodeBook
function SmallCodeBook = getSmallCodeBook(VQCodebook, numSpeakers, sizeCodeBook)
    SmallCodeBook = cell(1,numSpeakers);
    rows = log2(sizeCodeBook);
    powers = 1:1:rows;
    col = 2.^powers;

    for i=1:1:numSpeakers
        for j=1:1:rows
            for k=1:1:col(j)
                SmallCodeBook{i}{j,k} = VQCodebook{i}{j,k}(1,:); %get the first entry(centroid)

            end
        end
    end
end

%Store training vectors in addition to codewords in a full codebook
function FullCodeBook = getCodeBook(MFCCvectors, numSpeakers, sizeCodeBook)
loops = log2(sizeCodeBook); %how many times to loop?

FullCodeBook = cell(1, numSpeakers);

eps = 0.01; %splitting parameter

% BEGIN SPLIT

for i=1:1:numSpeakers
    sizeFrames = size(MFCCvectors{i},1);

    %calculate the initial centroid of the currenet speaker's codebook
    %centroid = [mean(x vals), mean (y vals)]
    initCentroid = [mean(MFCCvectors{i}(:,1)) mean(MFCCvectors{i}(:,2))];
    %initCentroid = initCentroid*2;
    
    %how many times are we splitting the codebook        
    FullCodeBook{i}{1}(1,:) = initCentroid * (1-eps);
    FullCodeBook{i}{2}(1,:) = initCentroid * (1+eps); 
        
    for doubles=1:1:loops
        numCWUpdate = doubles; %counter for calculating centroids
        
        sizeCB = size(FullCodeBook{i}, 2); %how many codevectors in the book now
        
        if(doubles>1)
            for kt=1:1:sizeCB
                ctr = 2*kt;
                
                %fetch the previous codebook centroid
                prevCentroid = FullCodeBook{i}{doubles-1, kt}(1,:);
                
                %split centroid into two again
                FullCodeBook{i}{doubles, ctr-1} = prevCentroid * (1-eps);
                FullCodeBook{i}{doubles, ctr} = prevCentroid * (1+eps);
                
            end
            
        end
        
        %update sizeCB
        sizeCB = size(FullCodeBook{i}, 2); %how many codevectors in the book now
        
        %keep track of how many times the length of the new split
        %is equal to the length of the previous split
        equalLength = 0; 

        %4 is a predefined value from trial and error. This will ensure
        %convergence of values for the centroids
        while(equalLength < 4)

            for k=1:1:sizeFrames
                currentVec = MFCCvectors{i}(k,:); %select the current MFCC vector to compare

                dist = zeros(1, sizeCB); %for holding the distances from training data to codebook

                for j=1:1:sizeCB
                    currentCBVec = FullCodeBook{i}{numCWUpdate, j}(1,:); %first row is the centroid

                    dist(j) = ( currentCBVec(1) - currentVec(1) )^2 + ...
                        ( currentCBVec(2) - currentVec(2) )^2;
                    
                    %dist(j) = sqrt(dist(j));

                end

             %assign currentVec to closest codeword
             closestCW = find(dist == min(dist)); %get index of closest codeword
             lenVector = size(FullCodeBook{i}{numCWUpdate, closestCW}, 1); %calculate next open position
             FullCodeBook{i}{numCWUpdate, closestCW}( lenVector+1, :) = currentVec;

            end        

            numSplits = size(FullCodeBook{i}, 1); %how many times have we split codebooks

            %now recalculate the centroids using assigned vectors
            for l=1:1:sizeCB            
                FullCodeBook{i}{numSplits+1,l} = [mean(FullCodeBook{i}{numSplits,l}(:,1)) ... 
                    mean(FullCodeBook{i}{numSplits,l}(:,2))];            
            end

            numCWUpdate = numCWUpdate + 1; %increment counter

            %make sure we have enough iterations
            if(numSplits > 1)
                %Check size of codebooks to know when to stop.
                numVecs = size( FullCodeBook{i}{numSplits,1}, 1 );
                numVecsPrev = size( FullCodeBook{i}{numSplits-1,1}, 1 );

                if(numVecs == numVecsPrev) 
                    equalLength = equalLength+1;
                end 

            end
        end %END OF WHILE LOOP
        
        %delete the iteration copies now that we don't need them.
        temp = cell(1,sizeCB);
        
        if(doubles==1)
                temp{1, 1} = FullCodeBook{i}{numSplits, 1}; %save new row too
                temp{1, 2} = FullCodeBook{i}{numSplits, 2}; %save new row too
        else
            %save the previous copies
            for nums=1:1:doubles
                for t=1:1:sizeCB
                    %build a cell array of the most recent copies
                    if(nums==doubles)
                        temp{nums, t} = FullCodeBook{i}{numSplits, t}; %save new row too
                    else
                        temp{nums, t} = FullCodeBook{i}{nums, t}; 
                    end
                end  
            end
        end
  
        %restore Codebook array
        FullCodeBook{i} = temp;
    end
end


end

%~~~~~~~~~~~~~~~Plotting Functions~~~~~~~~~~~~~~~~~~~~~~~%

%Plot all MFCC's of a speaker
function plotMFCCVectors(MFCCvectors, numSpeakers, VQDim)
figure;

%assemble title string for plots
titleStr = strcat('Acoustic Vectors per Speaker using [MFCC ',num2str(VQDim(1)), ... 
    ', MFCC ', num2str(VQDim(2)), ']');
title(titleStr);

hold on
colors = ['b', 'k', 'r', 'g', 'y', 'c', 'm', [0 .5 .75]];

for i=1:1:numSpeakers
    %assemble legend title
    legendTitle = strcat('Speaker ', num2str(i));
    
    %calculate which color to use
    index = mod(i, length(colors)) + 1;
    
    %scatter plot the x and y vectors [4, 16] MFCC columns
    scatter(MFCCvectors{i}(:,1), MFCCvectors{i}(:,2), 7, 'filled', colors(index), ...
        'DisplayName', legendTitle);
end
legend
%hold off
end

function plotMFCC(bank, numMelFilters, numSpeakers)
x = 1:1:numMelFilters;

    for k=1:1:numSpeakers
        left = mod(k,4); %calculate current plot number
        
        plotNum = left;
        
        if(left == 1) %if subplot figure is full, generate new figure
            figure;
        elseif(left == 0) 
            plotNum = 4; 
        end
       
        
        %fprintf("PLOTNUM:%i LEFT:%i\n", plotNum, left);
        
        subplot(4,1, plotNum);
        plot(x, bank{k});
        titleGraph = strcat('Speaker ', num2str(k));
        title(titleGraph);
        xlabel('MFCC #');
        ylabel('Magnitude');
    end

end

function plotTimeSignal(signal, fs)
    lenSignal = length(signal);
    timeAxis = 0 : (1/fs) : ( lenSignal/fs - 1/fs );

    figure;
    plot(timeAxis, signal, 'g');
    grid on;
    title('Signal Plot');
    xlabel('Time (s)');
    ylabel('Amplitude');

end

function plotFreqSignal(signal, fs, type)
    half = floor(length(signal)/2);
    
    %type ==1 means the signal is a pure FFT output
    if(type ==1)

        %remove data above nyquist limit = fs/2
        %freqAxis = linspace(0, fs/2, length(signal)/2);
        freqResolution = fs/length(signal);
        freqAxis = 0 : freqResolution : fs/2-freqResolution; %cut away nyquist limit

        
        %disp(half);
        figure;
        subplot(2,1,1);
        plot(freqAxis, abs(signal(1:half))/ length(signal), 'b'); %plot mag and normalized
        %cut away nyquist limit @ end/2 + 1

        grid on;
        title('FFT Plot');
        xlabel('Frequency (Hz)');
        ylabel('Magnitude (Normalized)');
        
        subplot(2,1,2);
        plot(freqAxis, 10*log10(abs(signal(1:half))), 'b');
        
        title('FFT Plot');
        xlabel('Frequency (Hz)');
        ylabel('dB');           
        
    elseif(type == 0) %means signal is from DC to NYQUIST
        figure;

        plot(linspace(0, (fs/2), length(signal)), 20*log10(signal));
        
        title('Mel Spectrum'), xlabel('Frequency (Hz)');
    end
end

%~~~~~~~~~~~~~~~DSP Functions~~~~~~~~~~~~~~~~~~~~~~~%

%Get periodogram/power spectral density up to nyquist limit (1:N/2)
function halfPSD = getHalfPSD(signal)
    N = length(signal);
    signal = signal(1:N/2);
    
    halfPSD = (1/(N)) * (abs(signal).^2);
    %PSD = (1/(fs*N)) * abs(signal).^2;     why is it fs*N instead of N?
    
    %conserve total power except for DC=1 and nyquist freq=(N/2+1) because ...
    %they do not occur twice, so skip index 1 and (N/2+1)!!
    halfPSD(2:end) = 2*halfPSD(2:end); %multiply by two because of FFT mirrored data

end

%Generate periodogram/power spectral density for full signal
function fullPSD = getFullPSD(signal)
    N = length(signal);
    
    fullPSD = (1/(N)) * (abs(signal).^2);
    %fullPSD = (abs(signal).^2);
    %PSD = (1/(fs*N)) * abs(signal).^2;     why is it fs*N instead of N?
    
    %conserve total power except for DC=1 and nyquist freq=(N/2+1) because ...
    %they do not occur twice, so skip index 1 and (N/2+1)!!
    %fullPSD(2:end) = 2*fullPSD(2:end); %multiply by two because of FFT mirrored data

end

%This function uses the DSP tool box, not needed because of the manual
%calcuations above
%{
function plotPeriodogram(signal,fs)
%uses hamming window to calculate periodogram, takes in TIME SIGNAL
[pxx,w] = periodogram(signal, hamming(length(signal)));
figure;
plot(w,10*log10(pxx));

end
%}

%Plot power spectral density
function plotPSD(signal, fs)
    N = length(signal);
    
    %DC COEFF IS AT 1. NYQUIST AT N/2+1
    signal = signal(1:N/2+1); 

    PSD = (1/(N)) * (abs(signal).^2);
    %PSD = (1/(fs*N)) * abs(signal).^2;     why is it fs*N instead of N?
    
    %conserve total power except for DC and nyquist freq because ...
      %they do not occur twice, so skip index 1 and N/2+1 !!
    PSD(2:end-1) = 2*PSD(2:end-1); 
  
    freq = 0 : fs/N : fs/2;
    
    figure;
    plot(freq,10*log10(PSD));
    grid on;
    title('Periodogram Using FFT');
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');

end

function MSR = calcMSR(sig1, sig2)
    MSR = sqrt(sum((sig1 - sig2).^2 )) / length(sig1);
end

%Frame signal into chunks
function chunkArray = getChunks(currentSpeaker, N, M)
        %Calculate length of currentSpeaker
    speakerLen = length(currentSpeaker);

    %Calculate the indexes of the start of each chunk for the speaker data
    numChunks = 1 : M : speakerLen;

    %How many samples were cutoff from completing another chunk?
    leftover = speakerLen - numChunks(end);

    %If leftover samples are less than the start of the next chunk
    %then the last index in numChunks is not needed because
    %we don't want to have an empty chunk!
    if(leftover < M)
        numChunks(end) = [];
        leftover = speakerLen - numChunks(end); %recalculate leftover 
    end

    %calculate number of chunks
    totalChunks = length(numChunks);

    %If the leftover samples are less than the amount needed for another
    %overlap
    if(leftover < N) 
            d_zeros = N - leftover; %how much zero pad do we need to complete the chunk
            %numChunks(end + 1) = numChunks(end) + M;    %add an extra chunk bound
            currentSpeaker = [currentSpeaker zeros(1,d_zeros)]; %zero pad the signal
    end

    %create array for storing chunks. rows = chunks. columns = samples
    speakerArr = zeros(totalChunks, N);

    %populate the array with each chunk to prepare for FFT
    for i=1:1: (totalChunks)
        speakerArr(i, 1:N) = currentSpeaker(numChunks(i) : numChunks(i)+N-1);
       % j = strcat( num2str(numChunks(i)),  ", ", num2str(numChunks(i)+N-1));
        %disp(j);
    end
    
    chunkArray = speakerArr;
end

%Read in WAV data for multiple speakers
function [SpeakerArr, fs] = getArr(numSpeakers, path)
SpeakerArr = cell(1, numSpeakers); %cell array to store speaker data

    %loop to fetch speaker data and store information
    for m=1:1:numSpeakers    
        %fetch samples and sampling frequency of every speaker
        [data, fs] = audioread(strcat(path,"s",num2str(m),".wav"));
        
        length = size(data);
        channels = length(2);
        
        %store speaker data transposed in a cell array
        data = data.';
        
        if(channels > 1)
            SpeakerArr{m} = data(1,:); %just use left channel
        else            
            SpeakerArr{m} = data;        
        end        
    end    
clearvars data
end

%tCreates a 3d cell array to store pages of speakers with chunks
function SpeechFrames = getFrames(array, numSpeakers, N, M)

SpeechFrames = cell(1, numSpeakers);

    for m=1:1:numSpeakers
        SpeechFrames{m} = getChunks(array{m}, N, M);
    end
end

%Calculates the Mel Frequency Cepstrum Coefficients of all speakers
function MFCC = getMFCC(arr, numSpeakers, N, numMelFilters, fs)
 
    %create hamming windows for each chunk
    window = hamming(N, 'symmetric'); %N point hamming window
    
    %window = hann(N);
    %wvtool(window);
    window = window.'; %transpose to have 1 row

    %create copy of our 3d array to apply hamming window
    chunk3d_window = arr;

    %storing FFT data
    chunk3d_FFT = arr;

    %storing power spectrum density data
    chunk3d_PSD = arr;

    %Mel Filtered Spectrum
    melFilters = melfb(numMelFilters, N, fs);
    
    %plot(linspace(0, (fs/2), N/2), melFilters')
    %%%%%%%%%%%%NEW METHOD
    chunk3d_MELMAX = cell(1, numSpeakers);
    chunk3d_MFCC_TEST = chunk3d_MELMAX;

    %FOR ONE SPEAKER GENERATE NUMMELFILTER MFCC
    MFCC = cell(1, numSpeakers);

    %apply hamming window to each chunk for every speaker
    for m=1 : 1 : numSpeakers

        %storage setup for MEL spectrum
        sizeRow = size(arr{m},1);

        chunk3d_MELMAX{m} = cell(1,sizeRow);

        %create zero array for mfcc's
        MFCC{m} = zeros(sizeRow, numMelFilters);

        %traverse chunk by chunk to apply hamming window
        for j=1 : 1 : size(arr{m},1)

            %windowed signal = signal * window
            chunk3d_window{m}(j, :) = chunk3d_window{m}(j, :) .* window;
            
            %FFT each chunk
            chunk3d_FFT{m}(j, :) = fft( chunk3d_window{m}(j, :) );

            % calculate PSD for each chunk using FFT chunk
            chunk3d_PSD{m}(j, :) = getFullPSD(chunk3d_FFT{m}(j, :)); 
            
            %plotPSD(chunk3d_PSD{m}(j,:), fs);
            %plotTimeSignal(chunk3d_window{m}(j, :), fs);
            
            %Take maximum value of each MEL FILTER BANK
            chunk3d_MELMAX{m}{j} = max( full(chunk3d_PSD{m}(j, 1:(N/2)) .* melFilters), [], 2);
            
            %plotTimeSignal(chunk3d_MELMAX{m}{j}.', fs);

            %stores numMelFilters MFCC's per FRAME
            chunk3d_MFCC_TEST{m}{j} = dct(chunk3d_MELMAX{m}{j});

            %compile each frame into an MFCC array
            MFCC{m}(j,:) = chunk3d_MFCC_TEST{m}{j}';        

        end
    end
end

%Returns the maximum value of the MFCC function above (for testing
%purposes)
function MaxMFCC = getMaxMFCC(arr, numSpeakers, N, numMelFilters, fs)
    %create hamming windows for each chunk
    window = hamming(N, 'symmetric'); %N point hamming window

    %wvtool(window);
    window = window.'; %transpose to have 1 row

    %create copy of our 3d array to apply hamming window
    chunk3d_window = arr;

    %storing FFT data
    chunk3d_FFT = arr;

    %storing power spectrum density data
    chunk3d_PSD = arr;

    %Mel Filtered Spectrum
    melFilters = melfb(numMelFilters, N, fs);

    %%%%%%%%%%%%NEW METHOD
    chunk3d_MELMAX = cell(1, numSpeakers);
    chunk3d_MFCC_TEST = chunk3d_MELMAX;

    %FOR ONE SPEAKER GENERATE NUMMELFILTER MFCC
    MaxMFCC = cell(1, numSpeakers);

    %apply hamming window to each chunk for every speaker
    for m=1 : 1 : numSpeakers

        %storage setup for MEL spectrum
        sizeRow = size(arr{m},1);

        chunk3d_MELMAX{m} = cell(1,sizeRow);

        %create zero array for mfcc's
        MaxMFCC{m} = zeros(sizeRow, numMelFilters);

        %traverse chunk by chunk to apply hamming window
        for j=1 : 1 : size(arr{m},1)

            %windowed signal = signal * window
            chunk3d_window{m}(j, :) = chunk3d_window{m}(j, :) .* window;
            
            %for removing DC coeff
            %chunk3d_window{i}(j, :) = detrend(chunk3d_window{i}(j, :)); 

            %FFT each chunk
            chunk3d_FFT{m}(j, :) = fft( chunk3d_window{m}(j, :) );

            %calculate PSD for each chunk using FFT chunk
            chunk3d_PSD{m}(j, :) = getFullPSD(chunk3d_FFT{m}(j, :)); 

            %%%%%%%%%%%%NEW METHOD%%%%%%%%%%%%%%%

            %take maximum value of each MEL FILTER BANK
            chunk3d_MELMAX{m}{j} = max( full(chunk3d_PSD{m}(j, 1:(N/2)) .* melFilters), [], 2);

            %stores numMelFilters MFCC's per FRAME
            chunk3d_MFCC_TEST{m}{j} = dct(chunk3d_MELMAX{m}{j});

            %compile each frame into an MFCC array
            MaxMFCC{m}(j,:) = chunk3d_MFCC_TEST{m}{j}';        

        end
        %lets take the max value of each MFCC and use it for our data
        MaxMFCC{m} = max(MaxMFCC{m}, [], 1);
    end
    
end

%~~~~~~~~~~~~~~~Open Source Functions~~~~~~~~~~~~~~~~~~~~~~~%

%Generates a Mel filter bank given the size
% Mel Filter Bank Function from:
% http://www.ifp.illinois.edu/~minhdo/teaching/speaker_recognition/
function m = melfb(p, n, fs)
% MELFB         Determine matrix for a mel-spaced filterbank
%
% Inputs:       p   number of filters in filterbank
%               n   length of fft
%               fs  sample rate in Hz
%
% Outputs:      x   a (sparse) matrix containing the filterbank amplitudes
%                   size(x) = [p, 1+floor(n/2)]
%
% Usage:        For example, to compute the mel-scale spectrum of a
%               colum-vector signal s, with length n and sample rate fs:
%
%               f = fft(s);
%               m = melfb(p, n, fs);
%               n2 = 1 + floor(n/2);
%               z = m * abs(f(1:n2)).^2;
%
%               z would contain p samples of the desired mel-scale spectrum
%
%               To plot filterbanks e.g.:
%
%               plot(linspace(0, (12500/2), 129), melfb(20, 256, 12500)'),
%               title('Mel-spaced filterbank'), xlabel('Frequency (Hz)');

f0 = 700 / fs;
fn2 = floor(n/2);

lr = log(1 + 0.5/f0) / (p+1);
%disp("lr");
%disp(lr);

% convert to fft bin numbers with 0 for DC term
bl = n * (f0 * (exp([0 1 p p+1] * lr) - 1));
%disp(bl);

b1 = floor(bl(1)) + 1;
b2 = ceil(bl(2));
b3 = floor(bl(3));
b4 = min(fn2, ceil(bl(4))) - 1;

pf = log(1 + (b1:b4)/n/f0) / lr;
fp = floor(pf);
pm = pf - fp;

r = [fp(b2:b4) 1+fp(1:b3)];
c = [b2:b4 1:b3] + 1;
v = 2 * [1-pm(b2:b4) pm(1:b3)];

%disp(v);

%m = sparse(r, c, v, p, 1+fn2);
m = sparse(r, c, v, p, fn2);
end