function NuT = compute_nusselt(Tfield, Twall, k, H, dy, ux)
% COMPUTE_NUSSELT Computes local Nusselt number along the bottom wall
%
% Inputs:
%   Tfield - temperature field [ny x nx]
%   Twall  - wall temperature
%   k      - thermal conductivity
%   H      - channel height
%   dy     - grid spacing in y
%   ux     - velocity profile [1 x ny]
%
% Output:
%   NuT    - local Nusselt number along the bottom wall [1 x nx]

    % Velocity-weighted mean temperature
    [ny, nx] = size(Tfield);
    Tmean = zeros(1,nx);
    y = linspace(0,H,ny);
    for i = 1:nx
        Tmean(i) = trapz(y, ux .* Tfield(:,i)') / trapz(y, ux);
    end

    % Heat flux at bottom wall (forward difference)
    q_bottom = k * (Twall - Tfield(2,:)) / dy;

    % Local Nusselt number
    NuT = (2*H/k) .* q_bottom ./ (Twall - Tmean);
end
