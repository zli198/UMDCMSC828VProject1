function [w,f,normgrad] = SNesterov(fun,gfun,label,w,bsz,kmax,tol)
    alpha=0.2;
    z=w;
    n = size(label,1);
    [n,~] = size(label);
    I = 1:n;
    f = zeros(kmax + 1,1);
    f(1) = fun(I,w);
    normgrad = zeros(kmax,1);
    for k = 1 : kmax
        Ig = randperm(n,bsz);
        b = gfun(Ig,w);
        normgrad(k) = norm(gfun(Ig,w));
        mu=1-3/(5+k-1);
        zt=z;
        z=w-alpha*b;
        w=(1+mu)*z-mu*zt;
        f(k + 1) = fun(I,w);
        if mod(k,100)==0
            fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
        end
        if normgrad(k) < tol
            break;
        end
    end
    fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end