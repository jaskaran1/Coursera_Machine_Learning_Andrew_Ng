function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
valset=[0.01 0.03 .1 .3 1 3 10 30];
C1=valset(1);
sigma1=valset(1);
model= svmTrain(X, y, C1, @(x1, x2) gaussianKernel(x1, x2, sigma1));
predictions=svmPredict(model,Xval);
pred_error=mean(double(predictions~=yval));
minerr=pred_error;
for i=1:length(valset)
    for j=1:length(valset)
    C1=valset(i);
    sigma1=valset(j);
    model= svmTrain(X, y, C1, @(x1, x2) gaussianKernel(x1, x2, sigma1));
    visualizeBoundary(X, y, model);
    predictions=svmPredict(model,Xval);
    pred_error=mean(double(predictions~=yval));
    %fprintf(['Prediction error=%f'],pred_error)
    if(minerr>pred_error)
        minerr=pred_error;
        C=C1;
        sigma=sigma1;
    end
    end
end
% =========================================================================

end