function [w,f,normgrad] = SNAG(fun,gfun,Y,w,bsz,kmax,tol)
    alpha=0.07;
    m=w;
    iter = 0;
    n = size(Y,1);
    [n,~] = size(Y);
    I = 1:n;
    f = zeros(kmax + 1,1);
    f(1) = fun(I,w);
    normgrad = zeros(kmax,1);
    
    while iter < kmax
        if iter~=0 && normgrad(iter) < tol
            break
        end
        iter = iter + 1;
        
        r_index = randperm(n,bsz);
        g = gfun(r_index,w);
        normgrad(iter) = norm(g);
        mu=1-3/(5+iter-1);
        m_pre=m;
        m=w-alpha*g;
        w=(1+mu) * m - mu*m_pre;
        f(iter + 1) = fun(I,w);
        if mod(iter,100)==0
            fprintf('iter = %d, f = %d, ||g|| = %d\n',iter,f(iter+1),normgrad(iter));
        end
        
    end
    fprintf('iter = %d, f = %d, ||g|| = %d\n',iter,f(iter+1),normgrad(iter));
end