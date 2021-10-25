function [w,f,normgrad] = SG(fun,gfun,label,w,bsz,kmax,tol)
    alpha=0.5;
    m0=100;
    l=0;
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
        w=w-alpha*b;
        f(k + 1) = fun(I,w);
        if mod(k,100)==0
            fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
        end
        if normgrad(k) < tol
            break;
        end
%         alpha=0.5/k;
        if (k>m0 && j>m0*2^l/l) || k==m0
            l=l+1;
            alpha=alpha/2;
            j=0;
        end
    end
    fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end