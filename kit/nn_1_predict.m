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

load('net.mat');

% Test the Network
outputs = net(inputs);
%errors = gsubtract(outputs,targets);
%performance = perform(net,targets,outputs)
op1 = outputs>0.5;
acc = sum(op1(1, :)==train_labels')/size(train_labels, 1)
 
%Test on test data
inputs_test = [word_test img_f_test];
inputs_test1 = inputs_test(1:89, :)';
outputs = net(inputs_test1);
%errors = gsubtract(outputs,targets);
%performance = perform(net,targets,outputs)
op1 = outputs>0.5;
acc = sum(op1(1, :)==act_test_labels')/size(act_test_labels, 1)

outputs = net(inputs_test');
op1 = outputs>0.5;
submit = op1(1, :)';
fileID = fopen('exp.txt','w');
fprintf(fileID, '%d\n', submit);
fclose(fileID);