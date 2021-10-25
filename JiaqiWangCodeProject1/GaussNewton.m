function [w,f,normgrad] = GaussNewton(Xtrain,label,w,kmax,tol)
    alpha=0.5;
    n = size(label,1);
    [n,~] = size(label);
    I = 1:n;
    f = zeros(kmax + 1,1);
    [r,J]=Res_and_Jac(Xtrain,label,w);
    f(1) = 0.5*sum(r.^2);
    normgrad = zeros(kmax,1);
    for k = 1 : kmax
        normgrad(k) = norm(J'*r);
        p=-(J'*J+1e-6*eye(size(J,2)))\(J'*r);
        w=w+alpha*p;
        [r,J]=Res_and_Jac(Xtrain,label,w);
        f(k + 1) = 0.5*sum(r.^2);
        if mod(k,100)==0
            fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
        end
        if normgrad(k) < tol
            break;
        end
    end
    fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end