# Machine_Learning_Course

This repository contains the exercises completed in the machine learning course from coursera.

## Aprendizado Supervisionado

### 1. Regressão Linear

  ####   a. Uma variável
A equação para o modelo, descrita na imagem abaixo, leva em consideração a entrada "x" e os parâmetros theta0 e theta1 para descobrir o objetivo, "y". 
A questão é como encontrar os melhores valores para os parâmetros theta0 e theta1. Para tanto, é preciso minimizar a função custo através do algoritmo gradiente descendente:
![image](https://user-images.githubusercontent.com/44439904/72576873-a2cd0980-38af-11ea-908b-11d8553e49b9.png)
_Figura 1 - Equações para aplicação modelo de regressão linear com uma vaiável_

Na prática, o código para calcular a função custo J pode ser visto abaixo: 
````

function J = computeCost(X, y, theta)
m = length(y); % number of training examples
J = 0;
h = ((X*theta) - y).^2;
S = sum(h(:));
J = S/(2*m);
end

````
Para facilitar o cógigo e prepará-lo para o caso de terem mais variáveis (x0,x1...) tratou-se X, theta e y como matrizes, acrescentando-se uma coluna de '1' a matriz X, que multiplica theta0, bem como na imagem com o modelo.

Em seguida, a função grandiente descendente recebe como parâmetros X, y, theta, alpha e o número de iterações:

````
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
````
obs: alpha pequeno demora a convergir, alpha muito alto talvez não decresça a função custo. Para que o algoritmo funcione, a atualização de theta0 e theta1 deve ser feita simultaneamente.

  #### b. Múltiplas variáveis
  Nesse caso, terão várias features no modelo, ampliando também a quantidade de theta. A solução do algoritmo Gradiente Descendente para minimizar a Função Custo também é válida, mas é interessante equiparar as escalas das variáveis quando elas forem eventualmente diferentes, utilizando feature scaling e/ou Normalização pela Média. Porém essa não é a única opção para encontrar os parâmetros, também é possível utilizar Equação Normal, uma solução mais rápida que não utiliza alpha ou iterações, e nem precisa de feature scaling. 
  ##### Mas quando usar Gradiente Descendente ou Equação Normal?
  Para um número n de features alto, é recomendado o gradiente descendente, porque equação normal realiza a inversa da matriz X, o que deixa o processo lento em casos de n elevado. Já para um n pequeno, a equação normal é mais recomendada, por não ter iterações e não utilizar alpha.

### 2. Regressão Logística

    Com a regressão logística queremos estimar variáveis discretas, e o que muda com relação a linear é o modelo:
    ````
    h = sigmoid(X*theta);
    ````

    Para evitar overfitting, pode-se reduzir o número de features ou usar regularização, que basicamente acrescenta um termo de regularização a função custo:
    ````
    regularterm = (lambda/m)*(theta(i));
    ````
    