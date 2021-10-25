function [w,f,gnorm] = Gauss_Newton(Xtrain, label, w, kmax, tol)
    f = nan(kmax+1,1);
    gnorm = nan(kmax+1,1);
    R_J = @(w) Res_and_Jac(Xtrain, label, w);
    [r, J] = R_J(w);
    f(1) = 0.5*sum(r.^2);
    gnorm(1) = norm(J'*r);
    gam = 0.9; % line search step factor
    jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
    eta = 0.5; % backtracking stopping criterion factor
    for k=1:kmax
        p = (J'*J+(1e-6).*eye(size(J,2)))\(-J'*r);
        ss = linesearch(w,p,R_J,eta,gam,jmax);
        step = ss*p;
        w = w + step;
        [r, J] = R_J(w);
        f(k+1) = 0.5*sum(r.^2);
        gnorm(k+1) = norm(J'*r);
        if mod(k,10)==0
            fprintf('iter = %d, f = %d, ||g|| = %d\n',k,f(k+1),gnorm(k+1));
        end
        if gnorm(k+1)<tol
            break
        end
    end
    fprintf('iter = %d, f = %d, ||g|| = %d\n',k,f(k+1),gnorm(k+1));
end

function [ss] = linesearch(x,p,func,eta,gam,jmax)
    ss = 1;
    [r,J] = func(x);
    f0 = 0.5*sum(r.^2);
    aux = eta*(J'*r)'*p;
    for j = 0 : jmax
        xtry = x + ss*p;
        [r1,~] = func(xtry);
        f1 = 0.5*sum(r1.^2);
        if f1 < f0 + ss*aux
            break;
        else
            ss = ss*gam;
        end
    end
end