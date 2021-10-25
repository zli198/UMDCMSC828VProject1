function [w,f,normgrad] = LevenbergMarquardt(Xtrain,label,w,kmax,tol)
    func=@(r) 0.5*sum(r.^2);
    f = zeros(kmax + 1,1);
    [r,J]=Res_and_Jac(Xtrain,label,w);
    RJ=@(w) Res_and_Jac(Xtrain,label,w);
    f(1) = func(r);
    normgrad = zeros(kmax,1);
    R=0.4;
    for k = 1 : kmax
        normgrad(k) = norm(J'*r);
        [w,R]=trustregion(func,RJ,w,r,J,R);
        [r,J]=Res_and_Jac(Xtrain,label,w);
        f(k + 1) = 0.5*sum(r.^2);
        if mod(k,100)==0
            fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
        end
        if normgrad(k) < tol
            break;
        end
    end
    fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k+1),normgrad(k));
end

function [w,R]=trustregion(func,RJ,w,r,J,R)
    Rmax=2;
    eta=0.2;
    p=findp(r,J,R);
    g=J'*r;
    B=J'*J;
    rnew=RJ(w+p);
    rho=(func(r)-func(rnew))/(-p'*g-0.5*p'*B*p);
    if rho<0.25
        R=0.25*R;
    elseif rho>=0.75 && norm(p)==R
        R=min(2*R,Rmax);
    end
    if rho>eta
        w=w+p;
    end
end

function p=findp(r,J,R)
    I=eye(size(J,2));
    B=J'*J+1e-6*eye(size(J,2));
    g=J'*r;
    p=-B\g;
    if norm(p)>R
        lam=1;
        while 1
            B1=B+lam*I;
            C=chol(B1);
            p=-C\(C'\g);
            np=norm(p);
            dd = abs(np - R);
            if dd < 1e-6
                break 
            end
            q = C'\p;
            nq = norm(q);
            lamnew = lam + (np/nq)^2*(np - R)/R; 
            if lamnew < 0
                lam = 0.5*lam; 
            else
                lam=lamnew;
            end
        end
    end
end