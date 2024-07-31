clear all;
close all;

load EEGdata.mat;


duration = 50;
overlap = 1;
time_windows = 1:overlap:size(EEGdata,2) - duration + 1;
Az = [];

% Bi-LSTM model
layers = [
    sequenceInputLayer(60)
    bilstmLayer(100, 'OutputMode', 'sequence')
    dropoutLayer(0.2)
    bilstmLayer(50, 'OutputMode', 'last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'GradientThreshold', 1, ...
    'ValidationFrequency', 5, ...
    'Verbose', false);


%segmentation
for i = 1:overlap:size(EEGdata,2)-duration+1
    win_start = i;
    win_end = win_start + duration - 1;
    disp(['Range from ', num2str(win_start), ' to ', num2str(win_end)]);
    
    win = EEGdata(:, win_start:win_end, :); 
    X = reshape(win, [60, duration, size(win,3)]);
    Y = categorical(stim); % Labels
    
     % shuffle data
    indices = randperm(size(X, 3));
    X = X(:, :, indices);
    Y = Y(indices);

    % cell imputs for bilstm
    X_cell = cell(1, size(X, 3));
    for j = 1:size(X, 3)
        X_cell{j} = squeeze(X(:, :, j));
    end
    k = 20; % folds
    cv = cvpartition(Y, 'KFold', k);
    aucScores = zeros(1, k);

    for j = 1:k
        % indices
        trainIdx = training(cv, j);
        valIdx = test(cv, j);

        XTrain = X_cell(trainIdx);
        YTrain = Y(trainIdx);
        XVal = X_cell(valIdx);
        YVal = Y(valIdx);

        net = trainNetwork(XTrain, YTrain, layers, options);

        [~, scores] = classify(net, XVal);

        Yval=grp2idx(YVal) - 1;
        
        [aucScore,~,~,~]=rocarea(scores(:, 2),Yval);
        %disp(aucScore)

        if aucScore < 0.5, aucScore = 0.5; end
        aucScores(j) = aucScore;
    end
    Az(i) = mean(aucScores);
end

[maxAz, idxMaxAz] = max(Az);

figure;
time_axis = 1:1:(size(EEGdata,2)-duration+1);
plot(time_axis,Az,'LineWidth',2);
title("Bi-LSTM 10-fold CV performance (50ms time window)");
xlabel("Time window centered at");
ylabel("Az");
ylim([0.4,1]);

hold on;
plot(time_axis(idxMaxAz), maxAz, 'ro');
