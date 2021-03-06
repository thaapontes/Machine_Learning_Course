function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
[qtd_x, qtd_y] = size(X);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
thetaenunciado = theta;
h = sigmoid(X*theta);
logone = log(h);
firstpart = (y')*(logone); 
secondpart = ((ones(m,1) - y)')*(log(ones(m,1)-h));
firstsum = sum(firstpart(:));
secondsum = sum(secondpart(:));
S = firstsum + secondsum;
theta(1) = 0;
square = (theta).^2;
soma = sum(square(:));
partial = (lambda/(2*m))*soma;
J = (-(S/m))+ partial;
for i=1:qtd_y
  theta = thetaenunciado;
  partone = (((sigmoid(X*theta) - y)')*X(:,i));
  parttwo = partone/m;
  theta(1) = 0;
  regularterm = (lambda/m)*(theta(i));
  %fprintf('%f \n', regularterm);
  grad(i) = parttwo + regularterm;
  %fprintf('%f \n', size(grad));
endfor



% =============================================================

end
