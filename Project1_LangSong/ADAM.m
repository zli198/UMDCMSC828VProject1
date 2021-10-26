function [w,f,normgrad] = ADAM(fun,gfun,Y,w,bsz,kmax,tol)
    npar = length(w);
    alpha = 0.1;
    beta_1 = 0.9;
    beta_2 = 0.999;
    epsilon = 1e-8;

    m = zeros(npar,1);
    v = zeros(npar,1);
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
        
        r_index = randperm(n,bsz);
        g = gfun(r_index,w);
        normgrad(iter+1) = norm(g);
        m = beta_1*m + (1-beta_1)*g;
        v = beta_2*v + (1-beta_2)*(g.*g);
        m_h = m/(1-beta_1^(iter+1));
        v_h = v/(1-beta_2^(iter+1));
        w = w - ((alpha*m_h)./(sqrt(v_h)+epsilon));
        
        iter = iter + 1;
        f(iter + 1) = fun(I,w);
        if mod(iter,100)==0
            fprintf('iter = %d, f = %d, ||g|| = %d\n',iter,f(iter+1),normgrad(iter));
        end
    end
    fprintf('iter = %d, f = %d, ||g|| = %d\n',iter,f(iter+1),normgrad(iter));
end