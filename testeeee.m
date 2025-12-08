% fem_cardiac_solve.m
% Complete Solver for Cardiac Electrophysiology (FitzHugh-Nagumo)
%
% Solves:
%   du/dt = Div(D Grad u) + f(u,z)
%   dz/dt = -epsilon * (k*u*(u-a-1) + z)
%
% Using Isotropic D = 1 initially.

clear; clc; close all;

%% 1. Setup and Mesh Loading
% Load the mesh data (assuming heart-sa0.mat is in path)
% If starting from scratch, one could use fem_make_mesh_2D
try
    load('heart-sa0.mat'); 
    % Ensure we have the right structure. 
    % Assuming the file contains 'Omega' or 'Omega_lin'.
    if exist('Omega_lin', 'var'), Omega = Omega_lin; end
catch
    warning('heart-sa0.mat not found. Creating a simple square mesh for testing.');
    % Fallback: Create 10x10 mesh on unit square
    e_m = fem_poly_2D(1, 'triangle'); % Linear triangles
    [x, T, Bndry] = fem_make_mesh_2D(10, [1 1], e_m);
    Omega.x = x; Omega.T = T; Omega.p = 1; Omega.e = e_m;
end

% Define parameters (Aliev-Panfilov / FHN)
k = 8.0;
a = 0.15;
eps_val = 0.01;
D = 1.0; % Isotropic diffusion (Constant)

%% 2. Finite Element Basis and Matrix Assembly
% Build basis functions at quadrature points
quad_rule = 5; 
Omega.e = fem_get_basis(Omega.p, quad_rule, 'triangle');

% Define Function Space (Standard Lagrange P1)
u_space = Omega; 

% Assemble Mass Matrix M: M_ij = ∫ φ_i φ_j dΩ
fprintf('Assembling Mass Matrix...\n');
M = fem_assemble_block_matrix(@element_mass_matrix, Omega, u_space, u_space);

% Note: Stiffness Matrix K is calculated implicitly within the nonlinear 
% residual function 'u_equation_residual' or can be pre-calculated here 
% if the solver was purely linear. We rely on the Newton solver for K.

%% 3. Initialization
n_nodes = size(Omega.x, 1);
U = zeros(n_nodes, 1); % Normalized Potential
Z = zeros(n_nodes, 1); % Recovery Variable

% Apply Stimulation (Initial Activation)
% Stimulate a small region (e.g., bottom left corner or defined Purkinje site)
stim_center = [min(Omega.x(:,1)), min(Omega.x(:,2))];
stim_radius = 0.2; % Adjust scale based on mesh units
dists = sqrt((Omega.x(:,1)-stim_center(1)).^2 + (Omega.x(:,2)-stim_center(2)).^2);
U(dists < stim_radius) = 1.0;

%% 4. Time Stepping Loop
dt = 0.1;          % Time step (ms)
T_final = 50.0;    % Simulation duration (ms)
n_steps = ceil(T_final / dt);

% Tolerances for Newton-Raphson
tol = 1e-6;
max_iter = 10;

fprintf('Starting Time Integration (%d steps)...\n', n_steps);

for n = 1:n_steps
    t = n * dt;
    
    % Store previous state
    U_old = U;
    Z_old = Z;
    
    % --- Step A: Solve for U^{n+1} (Nonlinear Newton-Raphson) ---
    % Equation: M(u_new - u_old)/dt + K*u_new - Source(u_new, z_old) = 0
    
    for iter = 1:max_iter
        % Pack variables for the residual function
        vars.dt = dt;
        vars.u.u = U;           % Current Guess u^{n+1, k}
        vars.u.t = u_space.T;
        vars.u_old.u = U_old;   % Previous Time Step u^n
        vars.u_old.t = u_space.T;
        vars.z.u = Z_old;       % Semi-implicit z^n
        vars.z.t = u_space.T;
        
        % 1. Compute Residual Vector R_u
        % Uses the provided u_equation_residual.m
        R_u = fem_assemble_block_residual(@u_equation_residual, Omega, u_space, vars);
        
        % Check Convergence
        res_norm = norm(R_u);
        if res_norm < tol
            break;
        end
        
        % 2. Compute Jacobian Matrix J_u (Perturbation)
        % Computes dR/du numerically
        J_u = fem_assemble_block_matrix_perturbation(@u_equation_residual, ...
                                                     Omega, u_space, u_space, vars);
        
        % 3. Update Solution
        delta_U = J_u \ R_u;
        U = U - delta_U;
    end
    
    % --- Step B: Solve for Z^{n+1} (Linear Decoupled) ---
    % Equation derived in afem-2.pdf: 
    % (1 + eps*dt)*M*Z^{n+1} = M*Z^n - eps*k*dt * F2(U^n)
    % where F2(u) = u(u - a - 1) projected onto basis
    
    % Assemble source vector F2 using U_old
    vars.u_old.u = U_old; 
    F2_vec = fem_assemble_block_residual(@element_f2_source, Omega, u_space, vars);
    
    % Linear Solve
    % LHS Matrix: A_z = (1 + eps*dt) * M
    % RHS Vector: b_z = M*Z_old - eps*k*dt * F2_vec
    
    lhs_mat = (1 + eps_val * dt) * M;
    rhs_vec = M * Z_old - eps_val * k * dt * F2_vec;
    
    Z = lhs_mat \ rhs_vec;
    
    % --- Visualization (Every 10 steps) ---
    if mod(n, 10) == 0
        fprintf('Step %d/%d, t = %.2f, Newton Iter: %d\n', n, n_steps, t, iter);
        fem_plot_solution(Omega, U, t);
        drawnow;
    end
end

%% Helper Functions (To be included at the end of script or separate files)

function [Me] = element_mass_matrix(e, testsp, trialsp, teste, triale, vars)
    % Computes local mass matrix: ∫ φ_i φ_j dΩ
    % w: quadrature weights, phi: basis functions
    w = teste.gw; 
    phi = teste.y; 
    % Outer product at each quadrature point sum
    Me = phi' * diag(w) * phi; 
end

function [Re] = element_f2_source(e, testsp, teste, vars)
    % Computes projection of nonlinear source for Z equation
    % F2 = ∫ φ_i * u^n * (u^n - a - 1) dΩ
    
    a = 0.15;
    
    % Get u^n values on this element
    u_indices = vars.u_old.t(e,:);
    u_nodal = vars.u_old.u(u_indices);
    
    % Interpolate u to quadrature points
    u_qp = teste.y * u_nodal;
    
    % Calculate nonlinear term at QPs
    f2_val = u_qp .* (u_qp - a - 1);
    
    % Integrate: Re_i = sum( w_q * phi_i(q) * f2(q) )
    Re = teste.y' * (teste.gw .* f2_val);
end

function fem_plot_solution(Omega, U, t)
    trisurf(Omega.T, Omega.x(:,1), Omega.x(:,2), U, 'EdgeColor', 'none');
    view(2); axis equal; colorbar; caxis([0 1]);
    title(sprintf('Potential u at t = %.2f', t));
    xlabel('x'); ylabel('y');
end