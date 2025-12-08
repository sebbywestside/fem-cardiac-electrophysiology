%  Cardiac Electrophysiology FEM Solve -- CRT Lead Optimization
%
%    This script implements the full time-stepping solution for the 
%    monodomain electrophysiology model with backward Euler time discretization
%    and Newton-Raphson iteration for the nonlinear potential equation.
%
%  by Sebastian [Your Name]
%  [Date]
%
clear all; close all;

% =========================================================================
% SETUP: Mesh, function spaces, and parameters
% =========================================================================

quad = 5;
etype = 'quadrilateral';

% Mesh parameters
NoElems1D = 20;  % Number of elements in each direction
h = [1 1] / NoElems1D;

% Time stepping parameters
dt = 0.1;           % Time step size (ms)
t_end = 500;        % Total simulation time (ms)
num_steps = ceil(t_end / dt);

% Nonlinear solver parameters
nonlinear_tol = 1e-6;
max_nr_iterations = 20;

% Problem parameters (cardiac-specific)
params.k = 8.0;
params.a = 0.15;
params.epsilon = 0.01;

% =========================================================================
% MESH GENERATION
% =========================================================================

% Make Domain function space (for mesh geometry)
Omega.name = 'domain';
Omega.dm = 2;
Omega.p = 1;
Omega.e = fem_get_basis(Omega.p, quad, etype);
[Omega.x, Omega.t, Omega.b] = fem_make_mesh_2D(NoElems1D, h, Omega.e);

% =========================================================================
% FUNCTION SPACE FOR MEMBRANE POTENTIAL (u)
% =========================================================================

u.name = 'u';
u.dm = 1;  % Scalar field
u.p = 1;   % Linear basis functions
u.e = fem_get_basis(u.p, quad, etype);
[u.x, u.t, u.b] = fem_make_mesh_2D(NoElems1D, h, u.e);

% Initialize solution vectors
u.u = zeros(size(u.x, 1), 1);      % Current time step u^{n+1}
u.u_prev = zeros(size(u.x, 1), 1); % Previous time step u^n

n_u = size(u.u, 1);

% =========================================================================
% FUNCTION SPACE FOR GATING VARIABLE (z)
% =========================================================================

z.name = 'z';
z.dm = 1;
z.p = 1;
z.e = fem_get_basis(z.p, quad, etype);
[z.x, z.t, z.b] = fem_make_mesh_2D(NoElems1D, h, z.e);

z.u = zeros(size(z.x, 1), 1);      % Current time step z^{n+1}
z.u_prev = zeros(size(z.x, 1), 1); % Previous time step z^n

n_z = size(z.u, 1);

% =========================================================================
% FIBER DIRECTION FIELD
% =========================================================================
% For cardiac tissue, fiber direction varies spatially
% Here: simplified spiral pattern (real models would load from mesh)
% For now: constant fiber direction [1, 0]^T for all elements

fiber_x = ones(size(u.x, 1), 1);  % x-component of fiber direction
fiber_y = zeros(size(u.x, 1), 1); % y-component of fiber direction
fiber_field = [fiber_x, fiber_y];

% Normalize (already unit vectors in this case)
fiber_norm = sqrt(fiber_x.^2 + fiber_y.^2);
fiber_field(:, 1) = fiber_field(:, 1) ./ fiber_norm;
fiber_field(:, 2) = fiber_field(:, 2) ./ fiber_norm;

% =========================================================================
% STIMULATION BOUNDARY CONDITIONS
% =========================================================================
% The mesh is created on domain [0, 1] × [0, 1]
% Define stimulation sites within this domain
%
% We'll stimulate a few nodes near the left side (like Purkinje entry points)
% and test a CRT lead position on the opposite side

% Define stim sites in normalized coordinates [0, 1] × [0, 1]
% Left side entry points (Purkinje-like):
stim_site_1 = [0.1, 0.4];   % Lower left
stim_site_2 = [0.1, 0.6];   % Upper left

% CRT lead candidate positions:
stim_site_3 = [0.9, 0.5];   % Right side

stim_sites = [stim_site_1; stim_site_2; stim_site_3];
stim_radius = 0.15;  % ~15% of domain width

% Identify nodes within stimulation radius
stim_nodes = [];
for i = 1:size(stim_sites, 1)
    dist = sqrt((u.x(:, 1) - stim_sites(i, 1)).^2 + ...
                (u.x(:, 2) - stim_sites(i, 2)).^2);
    stim_nodes = [stim_nodes; find(dist <= stim_radius)];
end
stim_nodes = unique(stim_nodes);

fprintf('Stimulation setup:\n');
fprintf('  - Number of stimulation sites: %d\n', size(stim_sites, 1));
fprintf('  - Number of stimulated nodes: %d\n', length(stim_nodes));
fprintf('  - Stimulation radius: %.4f\n\n', stim_radius);

% =========================================================================
% ACTIVATION TIME STORAGE
% =========================================================================
activation_time = inf(size(u.x, 1), 1);
u_threshold = 0.8;

% =========================================================================
% TIME STEPPING LOOP
% =========================================================================

fprintf('\n========================================\n');
fprintf('Cardiac Electrophysiology FEM Solver\n');
fprintf('========================================\n\n');

for n = 0:(num_steps - 1)
    
    t_current = n * dt;
    
    % =====================================================================
    % STIMULATION: Apply Dirichlet BC to stimulated nodes
    % =====================================================================
    % At each time step, check if stimulation pulse is active
    % Stimulate continuously during first 20 ms
    if t_current < 20.0
        u.u(stim_nodes) = 1.0;  % Fully depolarized
    else
        % After stimulation period, allow natural evolution
        % (but don't force repolarization to avoid artifacts)
    end
    
    % On first time step, verify stimulation was applied
    if n == 0
        fprintf('First time step verification:\n');
        fprintf('  - Number of stim nodes: %d\n', length(stim_nodes));
        if length(stim_nodes) > 0
            fprintf('  - Potential on stim nodes (before solve): %.4f\n', ...
                mean(u.u(stim_nodes)));
        else
            fprintf('  - WARNING: No stimulation nodes found!\n');
        end
    end
    
    % =====================================================================
    % NEWTON-RAPHSON ITERATION FOR u^{n+1}
    % =====================================================================
    
    residual_norm = 1.0;
    nr_iter = 0;
    
    fprintf('Time step n=%d (t=%.2f ms): ', n, t_current);
    
    while (residual_norm > nonlinear_tol) && (nr_iter < max_nr_iterations)
        
        % Initialize global residual and Jacobian
        R_global = zeros(n_u, 1);
        J_global = sparse(n_u, n_u);
        
        % ---------------------------------------------------------------
        % ASSEMBLY: Loop over elements
        % ---------------------------------------------------------------
        for e = 1:size(Omega.t, 1)
            
            % Compute element geometry and basis function derivatives
            [metrics] = fem_compute_metrics(Omega, e);
            [teste_u] = map_element_quantities(metrics, u.e);
            
            % Prepare variables structure for element routine
            vars.u = u;
            vars.u_prev = u.u_prev;    % Pass as vector, not structure
            vars.z = z;
            vars.dt = dt;
            vars.params = params;
            vars.fiber = fiber_field(u.t(e, :), :);  % Local fiber directions
            
            % Compute element residual and Jacobian
            [Re, Je] = cardiac_electro_u_eqn(e, u, teste_u, vars);
            
            % Assemble into global system
            global_indices = u.t(e, :);
            R_global(global_indices) = R_global(global_indices) + Re;
            J_global(global_indices, global_indices) = ...
                J_global(global_indices, global_indices) + Je;
        end
        
        % ---------------------------------------------------------------
        % APPLY BOUNDARY CONDITIONS (Dirichlet: u = u_stim on stim nodes)
        % ---------------------------------------------------------------
        for node_idx = stim_nodes'
            R_global(node_idx) = u.u(node_idx) - 1.0;  % u = 1 on stimulated nodes
            J_global(node_idx, :) = 0;
            J_global(node_idx, node_idx) = 1.0;
        end
        
        % ---------------------------------------------------------------
        % SOLVE NEWTON-RAPHSON: J * Δu = -R
        % ---------------------------------------------------------------
        delta_u = J_global \ (-R_global);
        
        % Update solution
        u.u = u.u + delta_u;
        
        % Check convergence
        residual_norm = norm(R_global, 2);
        nr_iter = nr_iter + 1;
        
    end
    
    fprintf('NR iter=%d, residual=%.2e\n', nr_iter, residual_norm);
    
    % =====================================================================
    % DETECT ACTIVATION TIMES
    % =====================================================================
    for node = 1:n_u
        if (u.u_prev(node) <= u_threshold) && (u.u(node) > u_threshold)
            if isinf(activation_time(node))
                activation_time(node) = t_current;
            end
        end
    end
    
    % =====================================================================
    % UPDATE GATING VARIABLE z (Local ODE solution)
    % =====================================================================
    % z^{n+1} = [z^n - Δt*ε*k*u^{n+1}*(u^{n+1} - a - 1)] / (1 + Δt*ε)
    
    for node = 1:n_z
        u_node = u.u(node);
        z_node = z.u_prev(node);
        
        numerator = z_node - dt * params.epsilon * params.k * u_node * ...
                           (u_node - params.a - 1.0);
        denominator = 1.0 + dt * params.epsilon;
        
        z.u(node) = numerator / denominator;
    end
    
    % =====================================================================
    % ADVANCE TO NEXT TIME STEP
    % =====================================================================
    u.u_prev = u.u;
    z.u_prev = z.u;
    
    % =====================================================================
    % OPTIONAL: Periodically output solution
    % =====================================================================
    if mod(n, 50) == 0
        fprintf('  Current u range: [%.4f, %.4f]\n', min(u.u), max(u.u));
    end
    
end

% =========================================================================
% POST-PROCESSING
% =========================================================================

fprintf('\n========================================\n');
fprintf('Solution Complete\n');
fprintf('========================================\n\n');

% Count activated nodes
num_activated = sum(~isinf(activation_time));

if num_activated > 0
    % Compute global activation time
    T_total = max(activation_time(~isinf(activation_time)));
    fprintf('Total activation time (T_total): %.2f ms\n', T_total);
else
    fprintf('WARNING: No nodes activated!\n');
    fprintf('Possible causes:\n');
    fprintf('  1. Stimulation sites not found (check stim_radius)\n');
    fprintf('  2. Solution not propagating (check parameters)\n');
    fprintf('  3. Activation threshold too high\n');
    T_total = NaN;
end

fprintf('Number of activated nodes: %d / %d\n', num_activated, n_u);
fprintf('\n');

% =========================================================================
% VISUALIZATION
% =========================================================================

figure('Position', [100, 100, 1200, 500]);

% Plot 1: Final membrane potential
subplot(1, 2, 1);
if strcmpi(u.e.etype, 'triangle')
    trisurf(u.t(:, 1:3), u.x(:, 1), u.x(:, 2), u.u, ...
        'Facecolor', 'interp', 'LineStyle', 'none');
    hold on;
    trisurf(u.t(:, 2:4), u.x(:, 1), u.x(:, 2), u.u, ...
        'Facecolor', 'interp', 'LineStyle', 'none');
else
    trisurf(u.t(:, 1:3), u.x(:, 1), u.x(:, 2), u.u, ...
        'Facecolor', 'interp', 'LineStyle', 'none');
end
colorbar;
title('Final Membrane Potential u (normalized)');
xlabel('x');
ylabel('y');
colormap(jet);
caxis([0, 1]);

% Plot 2: Activation time map (or potential if no activation)
subplot(1, 2, 2);
if num_activated > 0
    % Activation times are available
    act_time_plot = activation_time;
    act_time_plot(isinf(act_time_plot)) = nan;
    
    if strcmpi(u.e.etype, 'triangle')
        trisurf(u.t(:, 1:3), u.x(:, 1), u.x(:, 2), act_time_plot, ...
            'Facecolor', 'interp', 'LineStyle', 'none');
        hold on;
        trisurf(u.t(:, 2:4), u.x(:, 1), u.x(:, 2), act_time_plot, ...
            'Facecolor', 'interp', 'LineStyle', 'none');
    else
        trisurf(u.t(:, 1:3), u.x(:, 1), u.x(:, 2), act_time_plot, ...
            'Facecolor', 'interp', 'LineStyle', 'none');
    end
    colorbar;
    title('Activation Time Map (ms)');
    xlabel('x');
    ylabel('y');
    colormap(jet);
else
    % Show z (gating variable) if no activation
    if strcmpi(u.e.etype, 'triangle')
        trisurf(u.t(:, 1:3), u.x(:, 1), u.x(:, 2), z.u, ...
            'Facecolor', 'interp', 'LineStyle', 'none');
        hold on;
        trisurf(u.t(:, 2:4), u.x(:, 1), u.x(:, 2), z.u, ...
            'Facecolor', 'interp', 'LineStyle', 'none');
    else
        trisurf(u.t(:, 1:3), u.x(:, 1), u.x(:, 2), z.u, ...
            'Facecolor', 'interp', 'LineStyle', 'none');
    end
    colorbar;
    title('Gating Variable z (repolarization) - No activation detected');
    xlabel('x');
    ylabel('y');
    colormap(jet);
end

drawnow;

% =========================================================================
% HELPER FUNCTION: fem_compute_metrics
% =========================================================================
function [metrics] = fem_compute_metrics(Omega, elem)
    % Extract local node coordinates
    x = Omega.x(Omega.t(elem, :), 1);
    y = Omega.x(Omega.t(elem, :), 2);
    
    % Compute spatial coordinates at quadrature points
    metrics.x = Omega.e.y(:, :) * x;
    metrics.y = Omega.e.y(:, :) * y;
    
    % Compute Jacobian derivatives
    dxdxi1 = Omega.e.dy(:, :, 1) * x;
    dxdxi2 = Omega.e.dy(:, :, 2) * x;
    dydxi1 = Omega.e.dy(:, :, 1) * y;
    dydxi2 = Omega.e.dy(:, :, 2) * y;
    
    % Jacobian of mapping
    metrics.j = abs(dxdxi1 .* dydxi2 - dxdxi2 .* dydxi1);
    
    % Transformation matrix for gradients
    ja = dxdxi1 .* dydxi2 - dxdxi2 .* dydxi1;
    metrics.m11 = dydxi2 ./ ja;
    metrics.m12 = -dxdxi2 ./ ja;
    metrics.m21 = -dydxi1 ./ ja;
    metrics.m22 = dxdxi1 ./ ja;
end

% =========================================================================
% HELPER FUNCTION: map_element_quantities
% =========================================================================
function [teste] = map_element_quantities(metrics, e)
    teste = e;
    teste.gw = e.gw .* metrics.j;
    
    for i = 1:size(e.y, 2)
        teste.dy(:, i, 1) = metrics.m11 .* e.dy(:, i, 1) + ...
                            metrics.m21 .* e.dy(:, i, 2);
        teste.dy(:, i, 2) = metrics.m12 .* e.dy(:, i, 1) + ...
                            metrics.m22 .* e.dy(:, i, 2);
    end
end