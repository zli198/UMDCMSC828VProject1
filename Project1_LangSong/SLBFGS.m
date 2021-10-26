function [w,f,normgrad] = SLBFGS(fun,gfun,Y,w,bsz,kmax,tol)
    stepsize=0.1;
    Ng = bsz;
    Nh = 800;
    m0=100;
    m=5;
    freq = 10;
    n = size(Y,1);
    I = 1:n;
    f = zeros(kmax + 1,1);
    f(1) = fun(I,w);
    normgrad = zeros(kmax,1);
    
    % Stochastic gradient descent process
    r_index = randperm(n,Ng);
    g = gfun(r_index,w);
    w0 = w-stepsize*g;
    g0 = gfun(r_index,w0);
    s = w0 - w;
    y = g0 - g;
    rho = 1/(s'*y);
    
    for iter=1:kmax
        if iter<=m0
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
        
        r_index = randperm(n,Ng);
        g=gfun(r_index,w);
        p = finddirection(g,s,y,rho);
        w_pre = w;
        w = w + alpha*p;
        normgrad(iter) = norm(gfun(r_index, w));
        f(iter + 1) = fun(I,w);
        
        if mod(iter,freq)==0
            index_H = randperm(n,Nh);
            if size(s,1) == m
                s = s(:,2:m);
                y = y(:,2:m);
                rho = rho(2:m);
                s(:,m) = w-w_pre;
                y(:,m) = gfun(index_H,w) - gfun(index_H,w_pre);
                rho(m) = 1/(s(:,m)'*y(:,m));
            else
                s = [s, (w-w_pre)];
                y = [y, (gfun(index_H,w) - gfun(index_H,w_pre))];
                rho = [rho, 1/(s(:,end)'*y(:,end))];
            end
        end
        if mod(iter,100)==0
            fprintf('iter = %d, f = %d, ||g|| = %d\n',iter,f(iter+1),normgrad(iter));
        end
        if normgrad(iter)<tol
            ret = iter;
            break;
        end
        ret = iter;
    end
    fprintf('iter = %d, f = %d, ||g|| = %d\n',ret,f(ret+1),normgrad(ret));
    f = f(2:ret+1);
    normgrad = normgrad(1:ret);
end

%%
function p = finddirection(g,s,y,rho)
% input: g = gradient dim-by-1
% s = matrix dim-by-m, s(:,i) = x_{k-i+1}-x_{k-i}
% y = matrix dim-by-m, y(:,i) = g_{k-i+1}-g_{k-i}
% rho is 1-by-m, rho(i) = 1/(s(:,i)'*y(:,i))
m = size(s,2);
a = zeros(m,1);  
for i = 1 : m
    a(i) = rho(i)*s(:,i)'*g;
    g = g - a(i)*y(:,i);
end
gam = s(:,1)'*y(:,1)/(y(:,1)'*y(:,1)); % H0 = gam*eye(dim)
g = g*gam;
for i = m :-1 : 1
    aux = rho(i)*y(:,i)'*g;
    g = g + (a(i) - aux)*s(:,i);
end
p = -g;
end