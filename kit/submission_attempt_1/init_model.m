function model = init_model()
load('ens_logit.mat');
load('theta.mat');
load('svm1.mat');
load('indices.mat');

model.ens= ens;
model.theta=theta;
model.svm1=svm1;
model.indices=indices;
