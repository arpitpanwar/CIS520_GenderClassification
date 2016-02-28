word_train = importdata('train/words_train.txt');
word_test = importdata('test/words_test.txt');
%img_train = importdata('train/images_train.txt');
%img_test = importdata('test/images_test.txt');
%word_data = [word_train; word_test];
%[c_word, s_word, l_word] = pca(word_data);
train_labels = importdata('train/genders_train.txt');
act_test_labels = importdata('test/random.txt');
ip = word_train';
no_pc_word = 2600;
no_pc_img_feature = 6;

% img_test = importdata('test/images_test.txt');
% img_train = importdata('train/images_train.txt');
img_f_train = importdata('train/image_features_train.txt');
img_f_test = importdata('test/image_features_test.txt');
%submit = predict(model, Y);

% NEURAL NETWORK ATTEMPT
%wt = s_word(1:4998, :);
%wt = [wt(:, 1:2600) img_f_train];
inputs = [word_train img_f_train]';
targets = [train_labels, not(train_labels)]';
 
% Create a Fitting Network
hiddenLayerSize = 560;
net = patternnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%net.trainFcn = 'trainscg';

% Train the Network
[net,tr] = train(net,inputs,targets);

save net.mat
