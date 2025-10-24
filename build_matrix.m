function [A, b, x, y, ux] = build_matrix(nx, ny, L, H, rho, Gamma, Tin, Twall, Pe, scheme)
    % Build system matrix A, RHS b, grid x, y, ux
    % scheme = 'UD' or 'QUICK'
    
    dx = L / (nx-1);
    dy = H / (ny-1);
    umean = Pe * Gamma / (2*H*rho);
    umax = 3/2*umean;

    y = linspace(0,H,ny);
    x = linspace(0,L,nx);
    ux = umax*(1-(2*y/H - 1).^2);

    N = nx*ny;
    A = sparse(N,N);
    b = zeros(N,1);
    idx = @(i,j) (j-1)*nx + i;

    for j = 1:ny
        for i = 1:nx
            n = idx(i,j);
            
            if i == 1
                % Inlet: Dirichlet
                A(n,n) = 1; b(n) = Tin;
            elseif i == nx
                % Outlet: Neumann
                A(n,n) = 1; A(n,idx(i-1,j)) = -1; b(n) = 0;
            elseif j == 1 || j == ny
                % Walls: Dirichlet
                A(n,n) = 1; b(n) = Twall;
            else
                u = ux(j);
                aE = Gamma/dx^2;
                aN = Gamma/dy^2;
                aS = Gamma/dy^2;

                % Convective west coefficient
                if i == 2
                    % Second column: always UD
                    aW = Gamma/dx^2 + rho*u/dx;
                else
                    if strcmp(scheme,'UD')
                        aW = Gamma/dx^2 + rho*u/dx;
                    elseif strcmp(scheme,'QUICK')
                        % QUICK approximation (requires i-2)
                        % aW_quick = 6/8*rho*u/dx - 1/8*rho*u/dx previous neighbors
                        aW = Gamma/dx^2 + (6/8*rho*u/dx - 1/8*rho*u/dx); 
                    end
                end

                A(n,n) = aE + aW + aN + aS;
                A(n,idx(i+1,j)) = -aE;
                A(n,idx(i-1,j)) = -aW;
                A(n,idx(i,j+1)) = -aN;
                A(n,idx(i,j-1)) = -aS;
            end
        end
    end
end
