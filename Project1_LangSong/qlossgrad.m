function g = qlossgrad(I,Xtrain,label,w,lam)
aux = exp(-myquadratic(Xtrain,label,I,w));
a = -aux./(1+aux);
X = Xtrain(I,:);
d = size(X,2);
d2 = d^2;
y = label(I);
ya = y.*a;
qterm = X'*((ya*ones(1,d)).*X);
lterm = X'*ya;
sterm = sum(ya);
g = [qterm(:);lterm;sterm]/length(I) + lam*w;
end