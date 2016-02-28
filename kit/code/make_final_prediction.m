function predictions = make_final_prediction(model,X_test)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples

% Sample model
[coeff,score,latent] = pca(X_test);
predictions = score(:,1:250) * model.svmw;
predictions(predictions > 0.5) = 1;
predictions(predictions <= 0.5) = 0;

