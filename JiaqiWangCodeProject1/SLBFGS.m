function [w,f,normgrad] = SLBFGS(fun,gfun,label,w,kmax,tol)
    gam = 0.9; % line search step factor
    jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search 
    eta = 0.5; % backtracking stopping criterion factor
    alpha=0.007;
    m0=60;
    l=0;
    m=5;
    Ng=1301;
    NH=4000;
    M=10;
    n = size(label,1);
    [n,~] = size(label);
    I = 1:n;
    f = zeros(kmax + 1,1);
    f(1) = fun(I,w);
    normgrad = zeros(kmax,1);
    npar=length(w);
    s=zeros(npar,m);
    y=zeros(npar,m);
    rho=zeros(m);
    %first do one step of gradient descent
    Ig = randperm(n,Ng);
    g=gfun(Ig,w);
    wnew=w-alpha*g;
    gnew=gfun(Ig,wnew);
    s(:,1) = wnew - w;
    y(:,1) = gnew - g;
    rho(1) = 1/(s(:,1)'*y(:,1));
    H=s(:,1)'*y(:,1)/(y(:,1)'*y(:,1));
    wr=wnew;
    r=1;
    for k=1:kmax
        Ig = randperm(n,Ng);
        normgrad(k) = norm(gfun(Ig,w));
        sg=gfun(Ig,w);
        p=-H*sg;
%         [alpha,j] = linesearch(w,Ig,p,sg,fun,eta,gam,jmax);
%         if j == jmax
%             p = -g;
%             [alpha,~] = linesearch(w,Ig,p,sg,fun,eta,gam,jmax);
%         end
        wnew=w+alpha*p;
        f(k + 1) = fun(I,wnew);
        w=wnew;
        if mod(k,M)==0
            IH = randperm(n,NH);
            s=circshift(s,[0,1]);
            y=circshift(y,[0,1]);
            rho = circshift(rho,[0,1]);
            s(:,1) = wnew-wr;
            y(:,1) = gfun(IH,wnew) - gfun(IH,wr);
            rho(1) = 1/(s(:,1)'*y(:,1));
            wr = wnew;
            r=r+1;
            if r<m
                H=s(:,1)'*y(:,1)/(y(:,1)'*y(:,1));
                for j=1:r-1
                    V=(eye(npar)-rho(r-j)*y(:,r-j)*s(:,r-j)');
                    rh=1/(y(:,r-j)'*s(:,r-j));
                    H=V'*H*V+rh*(s*s');
                end
            else
                H=s(:,1)'*y(:,1)/(y(:,1)'*y(:,1));
                for j=1:m-1
                    V=(eye(npar)-rho(m-j)*y(:,m-j)*s(:,m-j)');
                    rh=1/(y(:,m-j)'*s(:,m-j));
                    H=V'*H*V+rh*(s*s');
                end
            end
        end
        if mod(k,100)==0
            fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
        end
        if normgrad(k)<tol
            break;
        end
        if (k>m0 && j>m0*2^l/l) || k==m0
            l=l+1;
            alpha=alpha/2;
            j=0;
        end
    end
    fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end

%%
function [a,j] = linesearch(x,I,p,g,func,eta,gam,jmax)
    a = 1;
    f0 = func(I,x);
    aux = eta*g'*p;
    for j = 0 : jmax
        xtry = x + a*p;
        f1 = func(I,xtry);
        if f1 < f0 + a*aux
            break;
        else
            a = a*gam;
        end
    end
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