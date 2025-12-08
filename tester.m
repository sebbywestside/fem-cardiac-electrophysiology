%% fem_solve_script_forward_implementation.m
%
% Cardiac Electrophysiology FEM Solver - FORWARD EULER / LINEARIZED
% Aliev-Panfilov Model with ISOTROPIC D=1 (for validation)
%
% Solves (linearized by evaluating nonlinear terms at u^n):
%   M * (u^{n+1} - u^n)/dt + K * u^{n+1} = M * f(u^n, z^n)
%   z^{n+1} = z^n + dt * g(u^n, z^n)
%
% where:
%   f(u,z) = ku(1-u)(u-a) - uz     (ionic current)
%   g(u,z) = -ε(ku(u-a-1) + z)     (gating dynamics)
%
% Rearranged for u:
%   (M/dt + K) * u^{n+1} = M/dt * u^n + M * f(u^n, z^n)
%
% This is LINEAR in u^{n+1} - no Newton-Raphson needed!
%
% Mesh: heart-sa0.mat
%
% by Sebastian (AFEM Project)

clear; close all; clc;

%% ========== PARAMETERS ==========

% Aliev-Panfilov model parameters
k = 8.0;
a = 0.15;
epsilon = 0.01;

% Diffusion coefficient (isotropic for now)
D = 1.0;

% Time stepping - use smaller dt for stability with explicit ionic term
dt = 0.1;          
T_final = 400;     
N_steps = ceil(T_final / dt);

% Stimulation parameters
stim_radius = 7.0;  
u_stim = 1.0;      

% Purkinje entry points on Γ₂ (right ventricular sites)
purkinje_sites = [83, 75;    % Site 1
                  19, 44;    % Site 2
                  124, 10];  % Site 3

% CRT lead positions to test on Γ₃
crt_sites = [42, 165;   % Option 1
             91, 179;   % Option 2
             138, 148]; % Option 3

fprintf('==============================================\n');
fprintf('  FORWARD EULER / LINEARIZED IMPLEMENTATION\n');
fprintf('==============================================\n\n');

%% ========== LOAD MESH ==========
fprintf('Loading mesh...\n');

mesh_file = 'heart-sa0.mat';
if exist(mesh_file, 'file')
    load(mesh_file);
    fprintf('Loaded %s\n', mesh_file);
else
    error('Mesh file %s not found!', mesh_file);
end

fprintf('\nMesh info:\n');
fprintf('  Nodes: %d\n', size(Omega.x, 1));
fprintf('  Elements: %d\n', size(Omega.t, 1));
fprintf('  Polynomial order: %d\n', Omega.p);

% Build element basis object
quad = 5;  % Lyness quadrature rule
Omega.e = fem_get_basis(Omega.p, quad, 'triangle');
fprintf('  Built basis object (quad rule = %d)\n', quad);

% Set domain properties
Omega.name = 'Omega';
Omega.dm = 2;

n_nodes = size(Omega.x, 1);
n_elems = size(Omega.t, 1);

%% ========== SETUP FUNCTION SPACES ==========

% u space (membrane potential)
u_space.name = 'u';
u_space.dm = 1;
u_space.p = Omega.p;
u_space.x = Omega.x;
u_space.t = Omega.t;
u_space.e = Omega.e;
u_space.b = Omega.b;

%% ========== ASSEMBLE GLOBAL MATRICES ==========
fprintf('\nAssembling global mass and stiffness matrices...\n');

% Mass matrix: M_ij = ∫ φ_i φ_j dΩ
M = fem_assemble_block_matrix(@element_mass_matrix, Omega, u_space, u_space);

% Stiffness matrix: K_ij = D * ∫ ∇φ_i · ∇φ_j dΩ
K = fem_assemble_block_matrix(@element_stiffness_matrix, Omega, u_space, u_space);

% Scale stiffness by diffusion coefficient
K = D * K;

% System matrix for implicit diffusion: A = M/dt + K
A = M / dt + K;

fprintf('  Mass matrix: %d x %d, nnz = %d\n', size(M,1), size(M,2), nnz(M));
fprintf('  Stiffness matrix: %d x %d, nnz = %d\n', size(K,1), size(K,2), nnz(K));

%% ========== IDENTIFY BOUNDARY AND STIMULATION NODES ==========
fprintf('\nIdentifying boundary nodes...\n');

bndry_nodes_patch2 = unique(Omega.b(Omega.b(:,end) == 2, 2:3));
bndry_nodes_patch3 = unique(Omega.b(Omega.b(:,end) == 3, 2:3));
bndry_nodes_patch2 = bndry_nodes_patch2(:);
bndry_nodes_patch3 = bndry_nodes_patch3(:);

fprintf('  Γ₂ boundary nodes: %d\n', length(bndry_nodes_patch2));
fprintf('  Γ₃ boundary nodes: %d\n', length(bndry_nodes_patch3));

fprintf('\nIdentifying Purkinje stimulation nodes...\n');

% Find boundary nodes on Γ₂ within stim_radius of Purkinje sites
stim_nodes_purkinje = [];
for i = 1:size(purkinje_sites, 1)
    site = purkinje_sites(i, :);
    for j = 1:length(bndry_nodes_patch2)
        node = bndry_nodes_patch2(j);
        dist = norm(Omega.x(node, :) - site);
        if dist <= stim_radius
            stim_nodes_purkinje = [stim_nodes_purkinje; node];
        end
    end
end
stim_nodes_purkinje = unique(stim_nodes_purkinje);
fprintf('  Purkinje stimulation nodes (Γ₂, within %.1f mm): %d\n', ...
        stim_radius, length(stim_nodes_purkinje));

% For baseline LBBB: only Purkinje stimulation
stim_nodes = stim_nodes_purkinje;

% Uncomment below to add CRT stimulation
% crt_site = crt_sites(1, :);  % Select which position (1, 2, or 3)
% stim_nodes_crt = [];
% for j = 1:length(bndry_nodes_patch3)
%     node = bndry_nodes_patch3(j);
%     dist = norm(Omega.x(node, :) - crt_site);
%     if dist <= stim_radius
%         stim_nodes_crt = [stim_nodes_crt; node];
%     end
% end
% stim_nodes_crt = unique(stim_nodes_crt);
% stim_nodes = unique([stim_nodes_purkinje; stim_nodes_crt]);

%% ========== APPLY DIRICHLET BC TO SYSTEM MATRIX ==========
% Modify A for Dirichlet BCs at stimulated nodes
% Set rows to identity, columns to zero (except diagonal)

A_bc = A;
for i = 1:length(stim_nodes)
    node = stim_nodes(i);
    A_bc(node, :) = 0;
    A_bc(node, node) = 1;
end

% Pre-factor the system matrix (LU decomposition for efficiency)
fprintf('\nFactoring system matrix with BCs...\n');
[L_sys, U_sys, P_sys] = lu(A_bc);
fprintf('  LU factorization complete\n');

%% ========== VISUALIZE MESH AND STIMULATION SITES ==========
figure(1); clf;
triplot(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), 'k-', 'LineWidth', 0.3);
hold on;

plot(Omega.x(bndry_nodes_patch2, 1), Omega.x(bndry_nodes_patch2, 2), ...
     'b.', 'MarkerSize', 8);
plot(Omega.x(bndry_nodes_patch3, 1), Omega.x(bndry_nodes_patch3, 2), ...
     'g.', 'MarkerSize', 8);
plot(Omega.x(stim_nodes, 1), Omega.x(stim_nodes, 2), ...
     'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(purkinje_sites(:,1), purkinje_sites(:,2), ...
     'b^', 'MarkerSize', 12, 'MarkerFaceColor', 'b', 'LineWidth', 2);
plot(crt_sites(:,1), crt_sites(:,2), ...
     'gs', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'LineWidth', 2);

title('Heart Mesh with Stimulation Sites');
xlabel('x_1 (mm)'); ylabel('x_2 (mm)');
legend('Mesh', 'Γ₂ (Purkinje boundary)', 'Γ₃ (CRT boundary)', ...
       'Stim nodes', 'Purkinje sites', 'CRT options', 'Location', 'best');
axis equal; grid on;
drawnow;

%% ========== INITIALIZE SOLUTION ==========
u_n = zeros(n_nodes, 1);       % u^n (current time level)
z_n = zeros(n_nodes, 1);       % z^n (current time level)

% Apply initial stimulation
u_n(stim_nodes) = u_stim;

% Activation time tracking
activation_time = inf(n_nodes, 1);
activation_threshold = 0.8;
activation_time(stim_nodes) = 0;

fprintf('\nApplied initial stimulation: u = %.2f at %d nodes\n', ...
        u_stim, length(stim_nodes));

%% ========== TIME STEPPING LOOP ==========
fprintf('\n========== STARTING TIME STEPPING ==========\n');
fprintf('dt = %.4f ms, T_final = %.1f ms, %d steps\n', dt, T_final, N_steps);
fprintf('Method: Forward Euler (ionic) + Backward Euler (diffusion)\n\n');

plot_interval = 100;

tic;
for n = 1:N_steps
    t_current = n * dt;
    
    %% === Step 1: Compute ionic current f(u^n, z^n) at old time level ===
    % f(u,z) = ku(1-u)(u-a) - uz
    f_ion = k * u_n .* (1 - u_n) .* (u_n - a) - u_n .* z_n;
    
    %% === Step 2: Form RHS and solve for u^{n+1} ===
    % RHS = M/dt * u^n + M * f(u^n, z^n)
    rhs = (M / dt) * u_n + M * f_ion;
    
    % Apply Dirichlet BC to RHS
    rhs(stim_nodes) = u_stim;
    
    % Solve: A * u^{n+1} = rhs
    % Using pre-factored LU: P*A = L*U, so A*x = b => x = U\(L\(P*b))
    u_new = U_sys \ (L_sys \ (P_sys * rhs));
    
    %% === Step 3: Update z^{n+1} using Forward Euler ===
    % dz/dt = g(u,z) = -ε(ku(u-a-1) + z)
    % z^{n+1} = z^n + dt * g(u^n, z^n)
    g_z = -epsilon * (k * u_n .* (u_n - a - 1) + z_n);
    z_new = z_n + dt * g_z;
    
    %% === Step 4: Track activation times ===
    newly_activated = (u_new > activation_threshold) & (activation_time == inf);
    activation_time(newly_activated) = t_current;
    n_activated = sum(activation_time < inf);
    
    %% === Step 5: Update for next time step ===
    u_n = u_new;
    z_n = z_new;
    
    % Enforce stimulation (Dirichlet BC)
    u_n(stim_nodes) = u_stim;
    
    %% === Progress output ===
    if mod(n, 100) == 0 || n == 1
        fprintf('t = %6.1f ms | u: [%.4f, %.4f] | z: [%.4f, %.4f] | activated: %d/%d (%.1f%%)\n', ...
                t_current, min(u_n), max(u_n), min(z_n), max(z_n), ...
                n_activated, n_nodes, 100*n_activated/n_nodes);
    end
    
    %% === Plot progress ===
    if mod(n, plot_interval) == 0
        figure(2); clf;
        trisurf(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), u_n, ...
                'Facecolor', 'interp', 'LineStyle', 'none');
        colorbar; caxis([0 1]);
        title(sprintf('Membrane Potential u at t = %.1f ms', t_current));
        xlabel('x_1 (mm)'); ylabel('x_2 (mm)');
        view(2); axis equal; colormap(jet);
        drawnow;
    end
    
    %% === Check if fully activated ===
    if n_activated == n_nodes
        fprintf('\n*** FULL ACTIVATION at t = %.2f ms ***\n', t_current);
        break;
    end
    
    %% === Stability check ===
    if any(isnan(u_n)) || any(isinf(u_n)) || max(abs(u_n)) > 10
        warning('Solution blowing up at t = %.2f ms! Try smaller dt.', t_current);
        break;
    end
end
elapsed_time = toc;

%% ========== RESULTS ==========
T_total = max(activation_time(activation_time < inf));
n_final_activated = sum(activation_time < inf);

fprintf('\n========== SIMULATION COMPLETE ==========\n');
fprintf('Elapsed wall time: %.2f seconds\n', elapsed_time);
fprintf('Total activation time: T_total = %.2f ms\n', T_total);
fprintf('Nodes activated: %d / %d (%.1f%%)\n', ...
        n_final_activated, n_nodes, 100*n_final_activated/n_nodes);

%% ========== FINAL VISUALIZATION ==========
figure(3); clf;
trisurf(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), u_n, ...
        'Facecolor', 'interp', 'LineStyle', 'none');
colorbar; caxis([0 1]);
title('Final Membrane Potential u');
xlabel('x_1 (mm)'); ylabel('x_2 (mm)');
view(2); axis equal; colormap(jet);

figure(4); clf;
act_time_plot = activation_time;
act_time_plot(activation_time == inf) = NaN;
trisurf(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), act_time_plot, ...
        'Facecolor', 'interp', 'LineStyle', 'none');
colorbar;
title(sprintf('Activation Time Map (T_{total} = %.1f ms)', T_total));
xlabel('x_1 (mm)'); ylabel('x_2 (mm)');
view(2); axis equal; colormap(jet);

figure(5); clf;
trisurf(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), z_n, ...
        'Facecolor', 'interp', 'LineStyle', 'none');
colorbar;
title('Final Gating Variable z');
xlabel('x_1 (mm)'); ylabel('x_2 (mm)');
view(2); axis equal; colormap(jet);

fprintf('\nDone! Check figures for visualization.\n');

%% ========== ELEMENT MATRIX FUNCTIONS ==========

function Me = element_mass_matrix(testsp, trialsp, teste, triale)
% element_mass_matrix -- Computes element mass matrix
%
%   M_ij = ∫ φ_i φ_j dΩ
%
% Inputs:
%   testsp, trialsp  - test and trial function spaces
%   teste, triale    - mapped basis functions at quadrature points
%
% Output:
%   Me - element mass matrix

    ne = size(testsp.t, 2);
    me = size(trialsp.t, 2);
    Me = zeros(testsp.dm * ne, trialsp.dm * me);
    
    for i = 1:ne
        for j = 1:me
            Me(i, j) = dot(teste.gw, teste.y(:,i) .* triale.y(:,j));
        end
    end
end

function Ke = element_stiffness_matrix(testsp, trialsp, teste, triale)
% element_stiffness_matrix -- Computes element stiffness matrix
%
%   K_ij = ∫ ∇φ_i · ∇φ_j dΩ
%
% Inputs:
%   testsp, trialsp  - test and trial function spaces
%   teste, triale    - mapped basis functions at quadrature points
%
% Output:
%   Ke - element stiffness matrix

    ne = size(testsp.t, 2);
    me = size(trialsp.t, 2);
    Ke = zeros(testsp.dm * ne, trialsp.dm * me);
    
    for i = 1:ne
        for j = 1:me
            % ∇φ_i · ∇φ_j = ∂φ_i/∂x * ∂φ_j/∂x + ∂φ_i/∂y * ∂φ_j/∂y
            grad_dot = teste.dy(:,i,1) .* triale.dy(:,j,1) + ...
                       teste.dy(:,i,2) .* triale.dy(:,j,2);
            Ke(i, j) = dot(teste.gw, grad_dot);
        end
    end
end