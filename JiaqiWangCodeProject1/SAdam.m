function [w,f,normgrad] = SAdam(fun,gfun,label,w,bsz,kmax,tol)
    alpha=0.1;
    beta1=0.9;
    beta2=0.999;
    eps=1e-8;
    npar=length(w);
    mo=zeros(npar,1);
    vo=zeros(npar,1);
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
        mo=beta1*mo+(1-beta1)*b;
        vo=beta2*vo+(1-beta2)*(b.*b);
        mot=mo/(1-beta1^k);
        vot=vo/(1-beta2^k);
        w=w-alpha*mot./(sqrt(vot)+eps);
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