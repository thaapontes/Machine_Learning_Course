function J = computeCost(X, y, theta)
m = length(y); % number of training examples
J = 0;
h = ((X*theta) - y).^2;
S = sum(h(:));
J = S/(2*m);
end
