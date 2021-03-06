function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X=[ones(m,1) X];%attached the bias parameter X0
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
a1=X';%a1 is 400x5000
Z2=Theta1*a1;%Z2 is a matrix of dimensions 25x5000 ie 25 is the number of 
             %cells in the 2nd layer and 5000 is the number of training examples
             %Thus,Z2 simultaneously holds all the values for the 2nd
             %layer for all the training examples
a2=sigmoid(Z2);%holds all the cells for 2nd layer for every training example after the sigmoid has been applied
a2=[ones(1,size(a2,2)); a2];%attached the bias parameter X0 for layer 2.This will act as X for theta2.Thus a2 is 26x5000
Z3=Theta2*a2;
a3=sigmoid(Z3);%a3 is a 10x5000 matrix,find max for all columns 
               %a3 is our h(theta) matrix
h=a3;%h has a size of Kxm
Y=zeros(m,num_labels);
for i=1:m
    Y(i,y(i))=1;
end
Y=Y';%Y has a size of Kxm
J=(1/m)*sum(sum(-Y.*log(h)-(1-Y).*log(1-h)));
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
delta_3=a3-Y;%delta_3 is of size KxM.So every column is a delta for the 3rd layer for a particular training example
Z2=[ones(1,size(Z2,2));Z2];
delta_2=Theta2'*delta_3.*sigmoidGradient(Z2);
delta_2=delta_2(2:end,:);%drop the 1st row of ones
del2=delta_3*a2';
del1=delta_2*a1';
Theta1_grad=1/m*del1;%no regularization
Theta2_grad=1/m*del2;%no regularization
% Part 3: Implement regularization with the cost function and gradients.
%Don't regularize the 1st column of Theta1 and Theta2

J=J+(lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2)))+sum(sum(Theta2(:,2:end).^2)));
S1=[zeros(size(Theta1,1),1) Theta1(:,2:end)];%since nothing is to be added in the first column
S2=[zeros(size(Theta2,1),1) Theta2(:,2:end)];%since nothing is to be added in the first column
Theta1_grad=Theta1_grad+(lambda/m)*S1;
Theta2_grad=Theta2_grad+(lambda/m)*S2;
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
