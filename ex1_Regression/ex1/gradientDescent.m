function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    h0 = ((X*theta) - y);
    S0 = sum(h0(:));
    J0 = S0/m;
    temp0 = theta(1) - (alpha*J0);
    
    h1 = (((X*theta) - y).*X(:,2));
    S1 = sum(h1(:));
    J1 = S1/m;
    temp1 = theta(2) - (alpha*J1);

    theta(1) = temp0;
    theta(2) = temp1;
    
    J_history(iter) = computeCost(X, y, theta);

end
end
