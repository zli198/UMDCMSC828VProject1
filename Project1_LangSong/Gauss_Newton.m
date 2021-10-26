function [w,f,normgrad] = Gauss_Newton(Xtrain,label,w,kmax,tol)
    func = @(r) 0.5*sum(r.^2);
    f = zeros(kmax+1,1);
    normgrad = zeros(kmax+1,1);
    R_J = @(w) Res_and_Jac(Xtrain, label, w);
    [r, J] = R_J(w);
    f(1) = func(r);
    normgrad(1) = norm(J'*r);
    gam = 0.9; % line search step factor
    jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
    eta = 0.5; % backtracking stopping criterion factor
    for iter=1:kmax
        p = (J'*J+(1e-6).*eye(size(J,2)))\(-J'*r);
        a = linesearch(w,p,R_J,eta,gam,jmax);
        w = w + a*p;
        [r, J] = R_J(w);
        f(iter+1) = 0.5*sum(r.^2);
        normgrad(iter+1) = norm(J'*r);
        if mod(iter,10)==0
            fprintf('iter = %d, f = %d, ||g|| = %d\n',iter,f(iter+1),normgrad(iter+1));
        end
        if normgrad(iter+1)<tol
            break
        end
    end
    fprintf('iter = %d, f = %d, ||g|| = %d\n',iter,f(iter+1),normgrad(iter+1));
end

function [a] = linesearch(x,p,func,eta,gam,jmax)
    a = 1;
    [r,J] = func(x);
    f0 = 0.5*sum(r.^2);
    aux = eta*(J'*r)'*p;
    for j = 0 : jmax
        xtry = x + a*p;
        [r1,~] = func(xtry);
        f1 = 0.5*sum(r1.^2);
        if f1 < f0 + a*aux
            break;
        else
            a = a*gam;
        end
    end
end