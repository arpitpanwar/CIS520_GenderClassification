function prediction = predict_log(theta, Xtest)

% This function returns the predicted label for the given X based on the
% highest probablity compared among each of the classifiers.

m = size(Xtest, 1);
prediction = zeros(size(Xtest, 1), 1);

% Add ones to the X data matrix to account for x0
Xtest = [ones(m, 1) Xtest];

prediction = (sigmoid(Xtest*theta') >= 0.5);