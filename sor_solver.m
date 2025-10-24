function [T_new, resHist, errHist, iter] = sor_solver(A, b, T_init, omega, tol, maxIter)
    %SOR_SOLVER Performs Successive Over-Relaxation iteration
    %
    % Inputs:
    %   A - system matrix
    %   b - right-hand side vector
    %   T_init - initial guess
    %   omega - relaxation factor
    %   tol - convergence tolerance
    %   maxIter - maximum number of iterations
    %
    % Outputs:
    %   T_new - converged solution
    %   resHist, errHist - convergence histories
    %   iter - number of iterations

    N = length(b);
    T_old = T_init;
    T_new = T_old;

    resHist = [];
    errHist = [];

    for k = 1:maxIter
        for n = 1:N
            diagA = A(n,n);
            if diagA == 0, continue; end
            sumNb = A(n,:) * T_new - diagA*T_new(n);
            T_new(n) = (1-omega)*T_new(n) + omega*(b(n)-sumNb)/diagA;
        end

        % Residual and relative error
        res = norm(A*T_new - b) / norm(diag(A).*T_new);
        err = norm(T_new - T_old) / norm(T_old);

        resHist(end+1) = res;
        errHist(end+1) = err;

        if res < tol && err < tol
            iter = k;
            fprintf('SOR converged (omega=%.2f) in %d iterations\n', omega, k);
            return;
        end

        T_old = T_new;
    end

    iter = maxIter;
    fprintf('SOR did not fully converge after %d iterations\n', maxIter);
end
