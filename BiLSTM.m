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

% training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'GradientThreshold', 1, ...
    'ValidationFrequency', 5, ...
    'Verbose', false);

for i = 1:overlap:size(EEGdata,2)-duration+1
    win_start = i;
    win_end = win_start + duration - 1;
    disp(['Range from ', num2str(win_start), ' to ', num2str(win_end)]);
    
    win = EEGdata(:, win_start:win_end, :); 
    X = reshape(win, [60, duration, 1, size(win,3)]);
    Y = categorical(stim);

    X = squeeze(X);
    X_cell = cell(1, size(X, 3));

    for j = 1:size(X, 3)
        X_cell{j} = X(:, :, j);
    end

    net = trainNetwork(X_cell, Y, layers, options);

    [~, scores] = classify(net, X_cell);

    y=grp2idx(Y) - 1;
    
    [AUC,~,~,~]=rocarea(scores(:, 2),y);

    if AUC < 0.5, AUC = 0.5; end
    Az(i) = AUC;

end

[maxAz, idxMaxAz] = max(Az);

figure;
time_axis = 1:1:(size(EEGdata,2)-duration+1);
plot(time_axis,Az,'LineWidth',2);
title("Bi-LSTM performance (50ms time window)");
xlabel("Time window centered at");
ylabel("Az");
ylim([0.4,1]);

hold on;
plot(time_axis(idxMaxAz), maxAz, 'ro');