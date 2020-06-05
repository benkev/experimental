function [X,F,Iters] = dfp(N, X, gradToler, fxToler, DxToler, MaxIter, myFx)
% Function dfp performs multivariate optimization using the
% Davidon-Fletcher-Powell method.
%
% Input
%
% N - number of variables
% X - array of initial guesses 
% gradToler - tolerance for the norm of the slopes
% fxToler - tolerance for function
% DxToler - array of delta X tolerances
% MaxIter - maximum number of iterations
% myFx - name of the optimized function
%
% Output
%
% X - array of optimized variables
% F - function value at optimum
% Iters - number of iterations
%

B = eye(N,N);

bGoOn = true;
Iters = 0;
% calculate initial gradient
grad1 =  FirstDerivatives(X, N, myFx);
grad1 = grad1';

while bGoOn

  Iters = Iters + 1;
  if Iters > MaxIter
    break;
  end

  S = -1 * B * grad1;
  S = S' / norm(S); % normalize vector S
  
  lambda = 1;
  lambda = linsearch(X, N, lambda, S, myFx);
  % calculate optimum X() with the given Lambda
  d = lambda * S;
  X = X + d;
  % get new gradient
  grad2 =  FirstDerivatives(X, N, myFx);

  grad2 = grad2';
  g = grad2 - grad1;
  grad1 = grad2;
  
  % test for convergence
  for i = 1:N
    if abs(d(i)) > DxToler(i)
      break
    end
  end
  
  if norm(grad1) < gradToler
    break
  end  
 
%  B = B + lambda * (S * S') / (S' * g) - ...
%      (B * g) * (B * g') / (g' * B * g);
  x1 = (S * S');
  x2 = (S * g);
  B = B + lambda * x1 * 1 / x2;
  x3 = B * g;
  x4 = B' * g;
  x5 = g' * B * g;
  B = B - x3 * x4' / x5;
end

F = feval(myFx, X, N);

% end

function y = myFxEx(N, X, DeltaX, lambda, myFx)

  X = X + lambda * DeltaX;
  y = feval(myFx, X, N);

% end

function FirstDerivX = FirstDerivatives(X, N, myFx)

for iVar=1:N  
  xt = X(iVar);
  h = 0.01 * (1 + abs(xt));
  X(iVar) = xt + h;
  fp = feval(myFx, X, N);
  X(iVar) = xt - h;
  fm = feval(myFx, X, N);
  X(iVar) = xt;
  FirstDerivX(iVar) = (fp - fm) / 2 / h;    
end

% end

function lambda = linsearch(X, N, lambda, D, myFx)

  MaxIt = 100;
  Toler = 0.000001;

  iter = 0;
  bGoOn = true;
  while bGoOn
    iter = iter + 1;
    if iter > MaxIt
      lambda = 0;
      break
    end  
   
    h = 0.01 * (1 + abs(lambda));
    f0 = myFxEx(N, X, D, lambda, myFx);
    fp = myFxEx(N, X, D, lambda+h, myFx);
    fm = myFxEx(N, X, D, lambda-h, myFx);
    deriv1 = (fp - fm) / 2 / h;
    deriv2 = (fp - 2 * f0 + fm) / h ^ 2;
    diff = deriv1 / deriv2;
    lambda = lambda - diff;
    if abs(diff) < Toler
      bGoOn = false;
    end
  end

% end
