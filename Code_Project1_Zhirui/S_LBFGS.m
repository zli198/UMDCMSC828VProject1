function [w,f,normgrad] = S_LBFGS(fun,gfun,label,w,bsz,kmax,tol)
    stepsize=0.1;
    Ng = bsz;
    Nh = 800;
    m0=100;
    m=5;
    Update_Frequency=10;
    n = size(label,1);
    I = 1:n;
    f = nan(kmax + 1,1);
    f(1) = fun(I,w);
    normgrad = nan(kmax,1);
    % Do stochastic gradient descent to find s_1, y_1
    index = randperm(n,Ng);
    s_grad= gfun(index,w);
    w_new = w-stepsize*s_grad;
    g_new=gfun(index,w_new);
    s = w_new - w;
    y = g_new - s_grad;
    rho = 1/(s'*y);
    klist=2:40;
    seq=idivide(int64(2.^klist),int64(klist));
    indexint=m0.*cumsum(seq);
    for k=1:kmax
        if k<=m0
            alpha=stepsize;
        elseif k<=2*m0
            alpha=stepsize/2;
        else
            ID=k-2*m0;
            kk=find(indexint>ID, 1, 'first' );
            alpha = stepsize/(2^(kk+1));
        end
        index = randperm(n,Ng);
        g=gfun(index,w);
        p = finddirection(g,s,y,rho);
        %disp(p)
        w_old=w;
        w=w+alpha*p;
        f(k + 1) = fun(I,w);
        normgrad(k) = norm(gfun(index, w));
        if mod(k,Update_Frequency)==0
            index_H = randperm(n,Nh);
            if size(s,1)==m
                s = s(:,2:m);
                y = y(:,2:m);
                rho = rho(2:m);
                s(:,m) = w-w_old;
                y(:,m) = gfun(index_H,w) - gfun(index_H,w_old);
                rho(m) = 1/(s(:,m)'*y(:,m));
            else
                s = [s, (w-w_old)];
                y = [y, (gfun(index_H,w) - gfun(index_H,w_old))];
                rho = [rho, 1/(s(:,end)'*y(:,end))];
            end
        end
        if mod(k,40)==0
            fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
        end
        if normgrad(k)<tol
            kout=k;
            break;
        end
        kout=k;
    end
    fprintf('k = %d, f = %d, ||g|| = %d\n',kout,f(kout+1),...
        normgrad(kout));
    f = f(2:kout+1);
    normgrad = normgrad(1:kout);
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
