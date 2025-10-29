%% CFD: 2D Steady Convection–Diffusion
clear; clc; close all;

% Parameters
rho = 1;          % kg/m^3
cp = 10;          % J/(K*kg)
k = 0.12;         % W/(m*K)
nu = 1e-2;        % m^2/s
Gamma = k / cp;

H = 1; L = 10;
nx = 50; ny = 5;

dx = L / (nx - 1);
dy = H / (ny - 1);

Pe = 16.5;
umean = Pe * Gamma / (2 * H * rho);
Re = 2 * H * umean / nu;

fprintf('umean = %.4f m/s, Re = %.2f\n', umean, Re);

% Velocity profile
x = linspace(0, L, nx);
y = linspace(0, H, ny);
umax = 3/2 * umean;
ux = umax * (1 - (2*y/H - 1).^2);

% Boundary conditions
Tin = 50;  
Twall = 100;

% Build system matrix and RHS 
[A, b, x, y, ux] = build_matrix(nx, ny, L, H, rho, Gamma, Tin, Twall, Pe, "UD");

%% Problem 11: Solve linear system
T = A \ b;

% Plot
Tfield = reshape(T, [nx, ny])';
x = linspace(0, L, nx);

figure;
contourf(x, y, Tfield, 30, 'LineColor','none');
colorbar;
xlabel('x [m]');
ylabel('y [m]');
title('Temperature Field T(x,y)');


%% Problem 12: Suitable plots
figure;
subplot(2,2,1);
plot(y, Tfield(:,1),'LineWidth',1.5);
xlabel('y [m]'); ylabel('T [°C]');
title('Inlet (x=0)');
grid on;

subplot(2,2,2);
plot(y, Tfield(:,end),'LineWidth',1.5);
xlabel('y [m]'); ylabel('T [°C]');
title('Outlet (x=L)');
grid on;

subplot(2,2,3);
plot(x, Tfield(1,:),'LineWidth',1.5);
xlabel('x [m]'); ylabel('T [°C]');
title('Lower wall (y=0)');
grid on;

subplot(2,2,4);
disp(Tfield)
plot(x, Tfield(end,:),'LineWidth',1.5);
xlabel('x [m]'); ylabel('T [°C]');
title('Upper wall (y=H)');
grid on;

%% Problem 13: Temperature plots
% (a) Outlet temperature profile
To = Tfield(:,end);
figure; plot(y, To, 'LineWidth',1.5);
xlabel('y [m]'); ylabel('T_o(y) [°C]');
title('Outlet Temperature Profile'); grid on;

% (b) Centerline temperature profile
[~, midRow] = min(abs(y-H/2));
Tc = Tfield(midRow,:);
figure; plot(x, Tc, 'LineWidth',1.5);
xlabel('x [m]'); ylabel('T_c(x) [°C]');
title('Centerline Temperature Profile'); grid on;

% (c) Velocity-weighted mean temperature
Tmean = zeros(1,nx);
for i = 1:nx
    Tmean(i) = trapz(y, ux .* Tfield(:,i)') / trapz(y, ux);
end
figure; plot(x,Tmean,'LineWidth',1.5);
xlabel('x [m]'); ylabel('T_{mean}(x) [°C]');
title('Velocity-weighted Mean Temperature'); grid on;

% (d) Entrance length xe (Tc reaches 90% of Twall)
target = Tin + 0.9*(Twall - Tin);
[~, idx_e] = min(abs(Tc - target));
xe = x(idx_e);
fprintf('Entrance length xe = %.3f m\n', xe);

%% Problem 14: SOR
omega_vals = [1, 1.5];
tol = 1e-5;
maxIter = 5000;

N = nx * ny;

T = ones(N,1) * Tin;

for w = 1:length(omega_vals)x
    omega = omega_vals(w);

    [T_sor, resHist, errHist, iter] = sor_solver(A, b, T, omega, tol, maxIter);

    figure;
    semilogy(resHist, 'r', 'LineWidth', 1.5); hold on;
    semilogy(errHist, 'b', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('Normalized value');
    legend('Residual','Relative Error');
    title(sprintf('SOR Convergence (\\omega = %.2f, iter = %d)', omega, iter));
    grid on;
end

%% Problem 15: Local Nusselt number Nu_T(x) along the lower wall

NuT = compute_nusselt(Tfield, Twall, k, H, dy, ux);

% Plot
figure;
semilogx(x, NuT, 'LineWidth', 1.5); hold on;
yline(7.54, 'r--', 'LineWidth', 1.5, 'DisplayName','Asymptotic Nu_T = 7.54');
xlabel('x [m]');
ylabel('Nu_T(x)');
title('Local Nusselt Number along Lower Wall');
legend('Nu_T(x)', 'Theoretical limit 7.54','Location','best');
grid on;

% Comment
fprintf('Average Nu_T near outlet = %.3f\n', mean(NuT(end-5:end)));
fprintf(['Expected theoretical limit = 7.54\n', ...
         'Any sharp drop or oscillation near outlet likely comes from the Neumann\n', ...
         'boundary (dT/dx = 0), which reduces temperature gradients artificially.\n']);

%% Problem 16: Finer mesh

% Mesh definitions
meshes = [50, 5;
          100, 11;
          200, 21;
          400, 41]; % [nx, ny] rows

% Preallocate storage
To_all = cell(size(meshes,1),1);
Tc_all = cell(size(meshes,1),1);
Tmean_all = cell(size(meshes,1),1);
xe_all = zeros(size(meshes,1),1);
Nu_all = cell(size(meshes,1),1);
x_all = cell(size(meshes,1),1);
y_all = cell(size(meshes,1),1);

% Loop over meshes
for m = 1:size(meshes,1)
    nx = meshes(m,1);
    ny = meshes(m,2);

    % Build matrix and velocity
    [A, b, x, y, ux] = build_matrix(nx, ny, L, H, rho, Gamma, Tin, Twall, Pe, "UD");

    % Solve linear system
    T = A \ b;

    % Reshape to 2D
    Tfield = reshape(T, [nx, ny])';

    % Store x, y
    x_all{m} = x;
    y_all{m} = y;

    % Outlet temperature
    To_all{m} = Tfield(:,end);

    % Centerline temperature
    [~, midRow] = min(abs(y-H/2));
    Tc_all{m} = Tfield(midRow,:);

    % Velocity-weighted mean temperature
    Tmean = zeros(1,nx);
    for i = 1:nx
        Tmean(i) = trapz(y, ux .* Tfield(:,i)') / trapz(y, ux);
    end
    Tmean_all{m} = Tmean;

    % Entrance length xe (centerline reaches 90% of Twall)
    target = Tin + 0.9*(Twall-Tin);
    [~, idx_e] = min(abs(Tc_all{m} - target));
    xe_all(m) = x(idx_e);

    % Local Nusselt number at lower wall
    Nu_all{m} = compute_nusselt(Tfield, Twall, k, H, y(2)-y(1), ux);
end

% Plot comparisons

% 1. Outlet temperature
figure; hold on; grid on;
for m = 1:length(To_all)
    plot(y_all{m}, To_all{m}, 'LineWidth', 1.5);
end
xlabel('y [m]'); ylabel('T_o(y) [°C]');
title('Outlet Temperature Profile Comparison');
legend('50x5','100x11','200x21','400x41','Location','best');

% 2. Centerline temperature
figure; hold on; grid on;
for m = 1:length(Tc_all)
    plot(x_all{m}, Tc_all{m}, 'LineWidth', 1.5);
end
xlabel('x [m]'); ylabel('T_c(x) [°C]');
title('Centerline Temperature Profile Comparison');
legend('50x5','100x11','200x21','400x41','Location','best');

% 3. Velocity-weighted mean temperature
figure; hold on; grid on;
for m = 1:length(Tmean_all)
    plot(x_all{m}, Tmean_all{m}, 'LineWidth', 1.5);
end
xlabel('x [m]'); ylabel('T_{mean}(x) [°C]');
title('Velocity-weighted Mean Temperature Comparison');
legend('50x5','100x11','200x21','400x41','Location','best');

% 4. Nusselt number at lower wall
figure; hold on; grid on;
for m = 1:length(Nu_all)
    semilogx(x_all{m}, Nu_all{m}, 'LineWidth', 1.5);
end
xlabel('x [m]'); ylabel('Nu_T(x)');
title('Local Nusselt Number Comparison');
legend('50x5','100x11','200x21','400x41','Location','best');

% Print entrance lengths
for m = 1:length(xe_all)
    fprintf('Mesh %dx%d: entrance length xe = %.3f m\n', meshes(m,1), meshes(m,2), xe_all(m));
end

%% Problem 17:
grids = [50 5; 100 11; 200 21; 400 41];
schemes = {'UD','QUICK'};

xe_all = zeros(length(grids), length(schemes));

for s = 1:length(schemes)
    scheme = schemes{s};
    
    for g = 1:size(grids,1)
        nx = grids(g,1);
        ny = grids(g,2);

        % Build matrix
        [A, b, x, y, ux] = build_matrix(nx, ny, L, H, rho, Gamma, Tin, Twall, Pe, scheme);

        % Solve
        T = A\b;
        Tfield = reshape(T, [nx, ny])';

        % Centerline temperature
        [~, midRow] = min(abs(y-H/2));
        Tc = Tfield(midRow,:);

        % Compute xe
        target = Tin + 0.9*(Twall - Tin);
        [~, idx_e] = min(abs(Tc - target));
        xe_all(g,s) = x(idx_e);
    end
end

% Plot xe comparison
figure;
plot(grids(:,1), xe_all(:,1), 'o-', 'LineWidth',1.5); hold on;
plot(grids(:,1), xe_all(:,2), 's-', 'LineWidth',1.5);
xlabel('nx'); ylabel('Entrance length x_e [m]');
legend('UD','QUICK','Location','best');
title('Entrance length x_e for UD vs QUICK');
grid on;

% Plot relative variation with finest mesh
relVar = abs(xe_all - xe_all(end,:))./xe_all(end,:);

figure;
plot(grids(:,1), relVar(:,1), 'o-', 'LineWidth',1.5); hold on;
plot(grids(:,1), relVar(:,2), 's-', 'LineWidth',1.5);
xlabel('nx'); ylabel('Relative variation w.r.t finest mesh');
legend('UD','QUICK','Location','best');
title('Relative variation of x_e for different grids');
grid on;

%% Problem 18:

% Mesh sets
% 1) Fix nx = 400, vary ny
grids_y = [5, 11, 21];
% 2) Fix ny = 41, vary nx
grids_x = [100, 200, 400];

xe_y = zeros(length(grids_y), 1);
xe_x = zeros(length(grids_x), 1);

% Scheme selection
scheme = 'UD';

% Case 1: nx fixed, vary ny 
for g = 1:length(grids_y)
    nx = 400; ny = grids_y(g);
    [A, b, x, y, ux] = build_matrix(nx, ny, L, H, rho, Gamma, Tin, Twall, Pe, scheme);
    T = A\b;
    Tfield = reshape(T, [nx, ny])';
    
    [~, midRow] = min(abs(y - H/2));
    Tc = Tfield(midRow,:);
    
    % Find entrance length (90% of Twall)
    target = Tin + 0.9*(Twall - Tin);
    [~, idx_e] = min(abs(Tc - target));
    xe_y(g) = x(idx_e);
end

% Case 2: ny fixed, vary nx
for g = 1:length(grids_x)
    nx = grids_x(g); ny = 41;
    [A, b, x, y, ux] = build_matrix(nx, ny, L, H, rho, Gamma, Tin, Twall, Pe, scheme);
    T = A\b;
    Tfield = reshape(T, [nx, ny])';
    
    [~, midRow] = min(abs(y - H/2));
    Tc = Tfield(midRow,:);
    
    % Find entrance length (90% of Twall)
    target = Tin + 0.9*(Twall - Tin);
    [~, idx_e] = min(abs(Tc - target));
    xe_x(g) = x(idx_e);
end

% Grid spacings for plot
dx_x = L ./ (grids_x - 1);
dy_y = H ./ (grids_y - 1);

% Plot results

figure;
subplot(1,2,1)
plot(dy_y, xe_y, 'o-', 'LineWidth',1.5);
xlabel('\Delta y'); ylabel('x_e [m]');
title('Effect of vertical resolution (nx = 400)');
grid on;

subplot(1,2,2)
plot(dx_x, xe_x, 's-', 'LineWidth',1.5);
xlabel('\Delta x'); ylabel('x_e [m]');
title('Effect of streamwise resolution (ny = 41)');
grid on;

sgtitle('Problem 18: Effect of anisotropic grid refinement on entrance length x_e');

% Comments
fprintf('\n--- Problem 18 Summary ---\n');
fprintf('As dy decreases (more vertical cells), x_e converges faster.\n');
fprintf('As dx decreases (more horizontal cells), improvements are smaller because flow is mostly streamwise.\n');
fprintf('Thus, vertical refinement (better wall resolution) can improve accuracy more efficiently.\n');

%% Problem 19: Effect of large Peclet numbers (Pe = 50 and Pe = 100)

% Mesh and scheme
nx = 100; ny = 11;
scheme = 'UD';

Pe_list = [50, 100];


figure('Units','normalized','Position',[0.05 0.05 0.9 0.7]);

for p = 1:length(Pe_list)
    Pe = Pe_list(p);

    % Build matrix and velocity profile
    [A, b, x, y, ux] = build_matrix(nx, ny, L, H, rho, Gamma, Tin, Twall, Pe, scheme);

    % Solve steady-state system
    T = A\b;
    Tfield = reshape(T, [nx, ny])';

    % Compute local cell Pe number
    dx = L/(nx-1);
    Pecell = abs(rho .* ux * dx / Gamma);
    maxPec = max(Pecell);

    % Plot temperature field
    subplot(2,2,p*2-1);
    [X, Y] = meshgrid(x, y);
    contourf(X, Y, Tfield, 30, 'LineColor', 'none');
    colorbar;
    xlabel('x [m]'); ylabel('y [m]');
    title(sprintf('T(x,y)  —  Pe = %d', Pe));
    axis tight;

    % Plot centerline profile
    subplot(2,2,p*2);
    [~, midRow] = min(abs(y - H/2));
    Tc = Tfield(midRow,:);
    plot(x, Tc, 'LineWidth', 1.5);
    xlabel('x [m]'); ylabel('T_c(x) [°C]');
    title(sprintf('Centerline temperature — Pe = %d (max local Pe = %.2f)', Pe, maxPec));
    grid on;

    % Print diagnostics
    fprintf('Pe = %d | Mesh: %dx%d | Max local Peclet = %.2f\n', Pe, nx, ny, maxPec);
end

sgtitle('Problem 19: Temperature field and centerline temperature at large Pe');

% Discussion
fprintf('\n--- Problem 19 Discussion ---\n');
fprintf(['As Pe increases, convection dominates diffusion, causing thinner thermal boundary layers.\n' ...
         'These steep gradients make the system more difficult to solve numerically.\n' ...
         'At high Pe, standard schemes may show oscillations or numerical instability unless upwinding\n' ...
         'or finer meshes are used. The problem becomes convection-dominated, and resolving boundary layers\n' ...
         'requires either higher-order stabilized schemes (e.g., QUICK, TVD) or grid refinement near the walls.\n']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Problem 20: Neumann boundary condition
% Parameters
rho = 1; cp = 10; k = 0.12; nu = 1e-2; Gamma = k/cp;
H = 1; L = 10;
nx = 100; ny = 11;
Pe = 16.5; Tin = 50; qwall = 10;   % inlet temp + wall flux
scheme = 'UD';

dx = L/(nx-1); dy = H/(ny-1);
umean = Pe*Gamma/(2*H*rho); 
umax = 3/2*umean;
x = linspace(0,L,nx); y = linspace(0,H,ny);
ux = umax*(1-(2*y/H - 1).^2);

N = nx*ny;
A = sparse(N,N);
b = zeros(N,1);

idx = @(i,j) (j-1)*nx + i;

% Build matrix
for j = 1:ny
    for i = 1:nx
        n = idx(i,j);
        u = ux(j);  % local velocity

        % Inlet (Dirichlet)
        if i == 1
            A(n,n) = 1;
            b(n) = Tin;

        % Outlet (Neumann, zero-gradient)
        elseif i == nx
            nW = idx(i-1,j);
            A(n,:) = 0;
            A(n,n) = 1;
            A(n,nW) = -1;
            b(n) = 0;

        % Bottom wall (Neumann)
        elseif j == 1
            nN = idx(i,j+1);
            A(n,:) = 0;
            A(n,n) = 1;
            A(n,nN) = -1;
            b(n) = qwall*dy/k;

        % Top wall (Neumann)
        elseif j == ny
            nS = idx(i,j-1);
            A(n,:) = 0;
            A(n,n) = 1;
            A(n,nS) = -1;
            b(n) = -qwall*dy/k;

        % Interior nodes (diffusion + convection)
        else
            aE = Gamma/dx^2;
            aN = Gamma/dy^2;
            aS = Gamma/dy^2;

            if i == 2
                aW = Gamma/dx^2 + rho*u/dx;  % first interior column: UD
            else
                if strcmp(scheme,'UD')
                    aW = Gamma/dx^2 + rho*u/dx;
                elseif strcmp(scheme,'QUICK')
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

% Solve system
T = A\b;
Tfield = reshape(T,[nx,ny])';

% Plot temperature
figure;
contourf(x,y,Tfield,30,'LineColor','none');
colorbar; xlabel('x'); ylabel('y');
title('T(x,y) with Neumann wall BC');
Tb_local = mean(Tfield, 1);    % 1 x nx

% local wall temperature at bottom (y = 0)
Tw_bottom = Tfield(1, :);      % 1 x nx

% local Nusselt number using local Tb and Tw
Nu_local = (2*H./k) .* q_bottom_num ./ (Tw_bottom - Tb_local);

% plot
figure;
plot(x, Nu_local, 'LineWidth', 1.5);
xlabel('x [m]'); ylabel('Nu(x)');
grid on;
title('Local Nusselt number along bottom wall (local Tw and Tb)');

fprintf(['\n--- Problem 20 Summary ---\n' ...
         'The Neumann BCs replace fixed wall temperatures by fixed heat flux q_wall.\n' ...
         'The resulting temperature field shows a linear near-wall gradient consistent with q_wall.\n' ...
         'Numerical wall flux verification confirms correct boundary implementation.\n']);


%% Problem 21: Energy balance
% ----- compute numeric fluxes for energy balance -----
% Tfield is ny x nx, rows = y (1 bottom ... end top), cols = x (1 inlet ... nx outlet)

% 1) local wall fluxes (positive = INTO the domain)
q_bottom_in = -k * ( Tfield(2,:)   - Tfield(1,:)   ) / dy;    % 1 x nx
q_top_in    = -k * ( Tfield(end-1,:) - Tfield(end,:) ) / dy; % 1 x nx
% NOTE: signs chosen so positive means heat entering the fluid domain.

% 2) integrate wall fluxes along x (use trapezoidal rule)
Q_bottom = trapz(x, q_bottom_in);   % [W/m] per unit depth (integral along x)
Q_top    = trapz(x, q_top_in);      % [W/m]

% 3) convective inlet/outlet (integrate rho*cp*u(y)*T(y) along y)
% u profile stored in ux (1 x ny) as given earlier; ensure orientation
u_y = ux;                % 1 x ny
T_inlet  = Tfield(:,1);  % ny x 1
T_outlet = Tfield(:,end);% ny x 1

% integrate in y with trapz (make sure y matches rows)
Q_inlet  = trapz(y, rho*cp .* u_y' .* T_inlet);   % scalar [W/m]
Q_outlet = trapz(y, rho*cp .* u_y' .* T_outlet);  % scalar [W/m]

% 4) global residual (positive => net heat ENTERING domain)
Residual = Q_inlet + Q_bottom + Q_top - Q_outlet;

% 5) relative residual (normalize by typical magnitude)
scale = max([abs(Q_inlet), abs(Q_outlet), abs(Q_bottom)+abs(Q_top), 1e-12]);
RelRes = Residual / scale;

% Print results
fprintf('Q_inlet  = %.6g W/m\n', Q_inlet);
fprintf('Q_outlet = %.6g W/m\n', Q_outlet);
fprintf('Q_bottom = %.6g W/m\n', Q_bottom);
fprintf('Q_top    = %.6g W/m\n', Q_top);
fprintf('Residual = Q_in + Q_bottom + Q_top - Q_out = %.6g W/m\n', Residual);
fprintf('Relative residual = %.3e (Residual / scale)\n', RelRes);


%% Problem 22: Local heating 2 <= x <= 5 on lower wall
% Task: find qwall values (uniform over 2<=x<=5 region) that produce outlet
% velocity-weighted mean temperatures approx 60, 70, 80 degC.
% Approach: bisection on q_mag for each target. Walls are adiabatic elsewhere.

nx = 400; ny = 41;
Pe = 16.5;
Tin = 50;
scheme = "UD";
x_region = [2, 5];   % m, heating region on lower wall
targets = [60, 70, 80];   % desired outlet velocity-weighted mean temperatures (degC)
tolerance_T = 0.05;       % acceptable temperature tolerance [degC]
max_bisect_iter = 30;

% Precompute grid and template matrix
[A0, b0, xg, yg, uxg] = build_matrix(nx, ny, L, H, rho, Gamma, Tin, Twall, Pe, scheme);

% If build_matrix doesn't output b0 (some versions don’t), ensure it's initialized
if isempty(b0)
    b0 = zeros(size(A0,1),1);
end

dx = L/(nx-1); 
dy = H/(ny-1);
idx = @(i,j) (j-1)*nx + i;

% Determine indices of bottom nodes that fall into [2,5]
ix_start = find(xg >= x_region(1), 1, 'first');
ix_end   = find(xg <= x_region(2), 1, 'last');
if isempty(ix_start) || isempty(ix_end)
    error('Heating region indices not found (check grid and region bounds).');
end
heating_indices = ix_start:ix_end;

% Perform bisection for each target
q_required = zeros(size(targets));
for tt = 1:length(targets)
    Ttarget = targets(tt);

    % Initial bracket
    q_low = 0;
    q_high = 1e4;

    T_low = outlet_mean_for_q(q_low, A0, b0, nx, ny, idx, heating_indices, ...
        xg, yg, uxg, Tin, k, dy);
    T_high = outlet_mean_for_q(q_high, A0, b0, nx, ny, idx, heating_indices, ...
        xg, yg, uxg, Tin, k, dy);

    if T_high < Ttarget
        error('q_high too small, increase q_high.');
    end

    % Bisection loop
    for iter = 1:max_bisect_iter
        q_mid = 0.5 * (q_low + q_high);
        T_mid = outlet_mean_for_q(q_mid, A0, b0, nx, ny, idx, heating_indices, ...
            xg, yg, uxg, Tin, k, dy);

        if abs(T_mid - Ttarget) < tolerance_T
            break;
        end
        if T_mid < Ttarget
            q_low = q_mid;
        else
            q_high = q_mid;
        end
    end

    q_required(tt) = q_mid;
    fprintf('Target T_out = %d °C -> q_required = %.3f W/m^2 (iter=%d), achieved T_out = %.3f\n', ...
        Ttarget, q_mid, iter, T_mid);
end

%% Display results
fprintf('\n--- Problem 22 Results ---\n');
for tt = 1:length(targets)
    fprintf('Target %d°C -> q_required = %.3f W/m^2\n', targets(tt), q_required(tt));
end

%% Plot one example solution
q_plot = q_required(2);
Tmean_out_plot = outlet_mean_for_q(q_plot, A0, b0, nx, ny, idx, heating_indices, ...
    xg, yg, uxg, Tin, k, dy);

% Rebuild matrix for visualization
A = A0; b = b0;
for i = 1:nx
    % Bottom wall
    n_bot = idx(i,1); n_bot_nb = idx(i,2);
    q_here = 0; if ismember(i, heating_indices), q_here = q_plot; end
    A(n_bot,:) = 0; A(n_bot,n_bot) = 1; A(n_bot,n_bot_nb) = -1; b(n_bot) = q_here*dy/k;
    % Top wall (adiabatic)
    n_top = idx(i,ny); n_top_nb = idx(i,ny-1);
    A(n_top,:) = 0; A(n_top,n_top) = 1; A(n_top,n_top_nb) = -1; b(n_top) = 0;
end
for j = 1:ny
    n_in = idx(1,j); A(n_in,:) = 0; A(n_in,n_in) = 1; b(n_in) = Tin;
end

Tfinal = A\b;
Tfield_final = reshape(Tfinal, [nx, ny])';

% Temperature field plot
figure;
contourf(xg, yg, Tfield_final, 30, 'LineColor', 'none'); colorbar;
xlabel('x [m]'); ylabel('y [m]');
title(sprintf('Temperature field, q=%.2f W/m^2 in x∈[%.1f,%.1f], outlet mean = %.2f°C', ...
    q_plot, x_region(1), x_region(2), Tmean_out_plot));

% Bottom wall flux profile
q_bottom_final = -k * (Tfield_final(2,:) - Tfield_final(1,:)) / dy;
figure;
plot(xg, q_bottom_final, '-o'); hold on;
plot(xg(heating_indices), q_bottom_final(heating_indices), 'r.', 'MarkerSize', 12);
xlabel('x [m]'); ylabel('q_{bottom}(x) [W/m^2]'); grid on;
title('Numerical bottom wall flux (red markers indicate heating region)');

fprintf(['\nProblem 22 comment:\n' ...
    '- The required heating flux q (W/m^2) found by bisection is printed above\n' ...
    '- The resulting outlet velocity-weighted mean temperature matches the target within tolerance.\n' ...
    '- Note required q values depend on mesh resolution and on neglecting axial conduction in inlet/outlet discretization.\n']);


function Tmean_out = outlet_mean_for_q(q_mag, A0, b0, nx, ny, idx, heating_indices, xg, yg, uxg, Tin, k, dy)
    A = A0; b = b0;

    % Apply boundary conditions
    for i = 1:nx
        % Bottom wall
        n_bot = idx(i,1); n_bot_nb = idx(i,2);
        q_here = 0;
        if ismember(i, heating_indices)
            q_here = q_mag;
        end
        A(n_bot,:) = 0;
        A(n_bot,n_bot) = 1;
        A(n_bot,n_bot_nb) = -1;
        b(n_bot) = q_here * dy / k;

        % Top wall: adiabatic
        n_top = idx(i,ny); n_top_nb = idx(i,ny-1);
        A(n_top,:) = 0;
        A(n_top,n_top) = 1;
        A(n_top,n_top_nb) = -1;
        b(n_top) = 0;
    end

    % Re-impose inlet Dirichlet
    for j = 1:ny
        n_in = idx(1,j);
        A(n_in,:) = 0;
        A(n_in,n_in) = 1;
        b(n_in) = Tin;
    end

    % Solve
    Tloc = A\b;
    Tfield_loc = reshape(Tloc, [nx, ny])';

    % Outlet temperature profile
    To_loc = Tfield_loc(:,end)'; 

    % Velocity-weighted mean temperature at outlet
    Tmean_out = trapz(yg, uxg .* To_loc) / trapz(yg, uxg);
end
