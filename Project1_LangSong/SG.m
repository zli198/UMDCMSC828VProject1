function [w,f,normgrad] = SG(fun,gfun,Y,w,bsz,kmax,tol)
    stepsize = 0.5; %initial stepsize
    m0=100;
    n = size(Y,1);
    I = 1:n;
    f = zeros(kmax + 1,1);
    f(1) = fun(I,w);
    normgrad = zeros(kmax,1);
    iter = 0;
    
    while iter<kmax
        if iter~=0 && normgrad(iter) < tol
            break
        end
        iter = iter+1;
        sampleindex = randperm(n,bsz);
        g = gfun(sampleindex,w);
        normgrad(iter) = norm(g);
        
        if iter<=m0 %for the first m0 iterations, use initial stepsize
            alpha=stepsize;
        elseif iter<=2*m0
            alpha=stepsize/2;
        else %if current iteration is greater than 2*m0 
            cur_iter = iter-2*m0;
            index = 2:40;
            factor = idivide(int64(2.^index),int64(index));
            round_length = m0.*cumsum(factor);
            round_index = find(round_length > cur_iter, 1, 'first' );
            alpha = stepsize/(2^(round_index+1));
        end
        
        w=w-alpha*g;
        f(iter+1) = fun(I,w);
        if mod(iter,100)==0
            fprintf('iter = %d, f = %d, ||g|| = %d\n',iter,f(iter+1),normgrad(iter));
        end
    end
    fprintf('iter = %d, f = %d, ||g|| = %d\n',iter,f(iter+1),normgrad(iter));
    f = f(2:iter+1);
    normgrad = normgrad(1:iter);
end