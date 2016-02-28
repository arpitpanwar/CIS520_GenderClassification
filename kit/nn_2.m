word_train = importdata('train/words_train.txt');
word_test = importdata('test/words_test.txt');
img_train = importdata('train/images_train.txt');
img_test = importdata('test/images_test.txt');
%word_data = [word_train; word_test];
%[c_word, s_word, l_word] = pca(word_data);
train_labels = importdata('train/genders_train.txt');
act_test_labels = importdata('test/random.txt');
ip = word_train';
no_pc_word = 2600;

% img_test = importdata('test/images_test.txt');
% img_train = importdata('train/images_train.txt');
img_f_train = importdata('train/image_features_train.txt');
img_f_test = importdata('test/image_features_test.txt');
%submit = predict(model, Y);

% NEURAL NETWORK ATTEMPT
%wt = s_word(1:4998, :);
%wt = [wt(:, 1:2600) img_f_train];
inputs = [word_train img_train img_f_train]';
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

% Test the Network
outputs = net(inputs);
%errors = gsubtract(outputs,targets);
%performance = perform(net,targets,outputs)
op1 = outputs>0.5;
acc = sum(op1(1, :)==train_labels')/size(train_labels, 1)
 
%Test on test data
inputs_test = [word_test img_test img_f_test];
inputs_test = inputs_test(1:89, :)';
outputs = net(inputs_test);
%errors = gsubtract(outputs,targets);
%performance = perform(net,targets,outputs)
op1 = outputs>0.5;
acc = sum(op1(1, :)==act_test_labels')/size(act_test_labels, 1)

% ATTEMPT TO FIT SVM 
% no = [];
% acc = [];
% i = 3;
% no_of_pc = 3000
% model = fitcsvm(s_word(1:4998, 1:no_of_pc), train_labels);
% test_labels = predict(model, s_word(1:4998, 1:no_of_pc));
% no(i) = no_of_pc;
% acc(i) =  sum(test_labels==train_labels)/size(test_labels, 1);
% 
% no_of_pc = [2600, 2650, 2700];
% for i=1:3
%     model = fitcsvm(s_word(1:4998, 1:no_of_pc(i)), train_labels);
%     test_labels = predict(model, s_word(4999:4999+size(act_test_labels, 1)-1, 1:no_of_pc(i)));
%     acc(i) =  sum(test_labels==act_test_labels)/size(test_labels, 1); 
% end