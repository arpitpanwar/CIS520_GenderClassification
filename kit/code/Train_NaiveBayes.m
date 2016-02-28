load ('../train/Train_Data.mat');

cp = cvpartition(Gender_Train,'KFold',10);

pooledTrain = horzcat(ImageFeatures_Train,Words_Train);
pooledTest = horzcat(ImageFeatures_Test,Words_Test);

[coeff,score,latent] = pca(vertcat(pooledTrain,pooledTest));

model_svm = fitcsvm(score(1:size(pooledTrain,1),1:1500),Gender_Train,'Standardize',true);
model_nb = fitcnb(score(1:size(pooledTrain,1),1:1500),Gender_Train);

svmResubErr = resubLoss(model_svm);
svmCV = crossval(model_svm, 'CVPartition',cp);
svmCVErr = kfoldLoss(svmCV);

nbGauResubErr = resubLoss(model_nb);
nbGauCV = crossval(model_nb, 'CVPartition',cp);
nbGauCVErr = kfoldLoss(nbGauCV);

svmLabels = predict(model_svm,score(size(pooledTrain,1)+1:size(score,1),1:500));
nbLabels = predict(model_nb,score(size(pooledTrain,1)+1:size(score,1),1:500));

finalLabels = (svmLabels + nbLabels)./2;
finalLabels(finalLabels > 0.5) = 1;
finalLabels(finalLabels <= 0.5) = 0;
