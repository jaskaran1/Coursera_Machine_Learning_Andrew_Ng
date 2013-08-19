function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
% 
m=size(X,1);%no of training examples
n=size(X,2);%no of training parameters
M=mean(X,1);%mean along the columns for every parameter
SIG=std(X,1);%s.d along the columns
%M and SIG have 1xn as the dimensions
mu=M(1)*ones(m,1);%initialize mu.mu has mx1 dimension ie it contains the mean for the first parameter in all the m cells 
sigma=SIG(1)*ones(m,1);%initialize sigma has mx1 dimension
for i=2:n
 mi=M(i)*ones(m,1);%m1 is column vector mx1 containing mu1
 si=SIG(i)*ones(m,1);
 mu=[mu mi];
 sigma=[sigma si];
end
%finally mu is mxn matrix.1st col has mu1,2nd has mu2 and so on
%similarly for sigma
X_norm=(X-mu)./sigma;
% ============================================================

end