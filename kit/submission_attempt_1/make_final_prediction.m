function predictions = make_final_prediction(model,X_test,X_train)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples


for i=1:size(X_test,1)
  cur_row=X_test(i,5001:35000);
  cur_img=reshape(cur_row,[100 100 3]);
  cur_img=rgb2gray(uint8(cur_img));
  img_features_test(i,:)=extractHOGFeatures(cur_img);
end
chosen_img_test=img_features_test(:,model.indices.topHog);
X=horzcat(X_train(:,model.indices.topwords),model.indices.chosen_image,X_train(:,35001:35007));
Xtest=horzcat(X_test(:,model.indices.topwords),chosen_img_test,X_test(:,35001:35007));
Y1=predict(model.ens,Xtest);
prediction=predict_log(model.theta,Xtest);
Y2=prediction(:,1);
Ktest=kernel_intersection(X,Xtest);
Ytest=zeros([size(X_test,1) 1]);
[yhat acc vals] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model.svm1);
Ymerger=horzcat(Y1,Y2,yhat);
predictions=mode(Ymerger,2);

