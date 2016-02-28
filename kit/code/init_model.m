function model = init_model()

load('Model.mat');
model.svmw = model_svm.W;


% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];
