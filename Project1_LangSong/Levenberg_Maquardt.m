function [w,f,normgrad] = Levenberg_Maquardt(Xtrain,label,w,kmax,tol)
    func = @(r) 0.5*sum(r.^2);
    d_max = 2;
    d_min = 0.8;
    d0 = (rand(1)*(d_max-d_min))+d_min;
    eta = 0.2;
    f = zeros(kmax+1,1);
    normgrad = zeros(kmax+1,1);
    R_J = @(w) Res_and_Jac(Xtrain, label, w);
    [r, J] = R_J(w);
    f(1) = func(r);
    normgrad(1) = norm(J'*r);
    
    for k=1:kmax
        [d0,w] = trustregion(d0, R_J, w, eta, d_max);
        [r, J] = R_J(w);
        f(k+1) = 0.5*sum(r.^2);
        normgrad(k+1) = norm(J'*r);
        if mod(k,10)==0
            fprintf('iter = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k+1));
        end
        if normgrad(k+1)<tol
            break
        end
    end
    fprintf('iter = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k+1));
end

function [R,w] = trustregion(currentRadius, func, w, eta, max_R)
    p = findp(func, w, currentRadius);
    wtry = w+p;
    [r0, J0] = func(w);
    [r1, ~] = func(wtry);
    f0 = 0.5*sum(r0.^2);
    g0 = J0'*r0;
    B = J0'*J0;
    f1 = 0.5*sum(r1.^2);
    rho = (f0-f1)/(-g0'*p-0.5.*p'*B*p);
    if rho<1/4
        R = 0.25*currentRadius;
    elseif rho>3/4 && norm(p)==currentRadius
        R = min(max_R, 2*currentRadius);
    else
        R = currentRadius;
    end
    if rho>eta
        w = wtry;
    else
        w=w;
    end
end

function [p] = findp(func, w, R)
    [r,J] = func(w);
    I = eye(size(J,2));
    B = J'*J + (1e-6)*I;
    g = J'*r;
    pstar = -B\g; % unconstrained minimizer
    if norm(pstar) <= R
        p = pstar;
    else % solve constrained minimization problem
        lam = 1; % initial guess for lambda
        while 1
            B1 = B + lam*I;
            C = chol(B1); % do Cholesky factorization of B
            p = -C\(C'\g); % solve B1*p = -g
            np = norm(p);
            dd = abs(np - R); % R is the trust region radius
            if dd < 1e-6
                break
            end
            q = C'\p; % solve C^\top q = p
            nq = norm(q);
            lamnew = lam + (np/nq)^2*(np - R)/R;
            if lamnew < 0
                lam = 0.5*lam;
            else
                lam = lamnew;
            end
        end
    end
end