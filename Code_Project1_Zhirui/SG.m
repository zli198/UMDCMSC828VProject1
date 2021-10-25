function [w,f,normgrad] = SG(fun,gfun,label,w,bsz,kmax,tol)
    stepsize = 0.5;
    m0=100;
    n = size(label,1);
    I = 1:n;
    f = nan(kmax + 1,1);
    f(1) = fun(I,w);
    normgrad = nan(kmax,1);
    iter = 0;
    klist=2:40;
    seq=idivide(int64(2.^klist),int64(klist));
    indexint=m0.*cumsum(seq);
    while iter<kmax
        if iter~=0 && normgrad(iter) < tol
            break
        end
        iter = iter+1;
        index = randperm(n,bsz);
        grad_new = gfun(index,w);
        normgrad(iter) = norm(grad_new);
        if iter<=m0
            alpha=stepsize;
        elseif iter<=2*m0
            alpha=stepsize/2;
        else
            ID=iter-2*m0;
            kk=find(indexint>ID, 1, 'first' );
            alpha = stepsize/(2^(kk+1));
        end
        w=w-alpha*grad_new;
        f(iter+1) = fun(I,w);
        if mod(iter,100)==0
            fprintf('iter = %d, f = %d, ||g|| = %d\n',...
                iter,f(iter+1),normgrad(iter));
        end
    end
    fprintf('iter = %d, f = %d, ||g|| = %d\n',iter,f(iter+1),...
        normgrad(iter));
    f = f(2:iter+1);
    normgrad = normgrad(1:iter);
end