function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%X has a size of 5000x400 m=5000 and n=400
X=[ones(m,1) X];%attached the bias parameter X0
Z2=Theta1*X';%Z2 is a matrix of dimensions 25x5000 ie 25 is the number of 
             %cells in the 2nd layer and 5000 is the number of training examples
             %Thus,Z2 simultaneously holds all the values for the 2nd
             %layer for all the training examples
a2=sigmoid(Z2);%holds all the cells for 2nd layer for every training example after the sigmoid has been applied
a2=[ones(1,size(a2,2)); a2];%attached the bias parameter X0 for layer 2.This will act as X for theta2.Thus a2 is 26x5000
Z3=Theta2*a2;
a3=sigmoid(Z3);%a3 is a 10x5000 matrix,find max for all columns 
[Y,I]=max(a3);
p=I';





% =========================================================================


end
