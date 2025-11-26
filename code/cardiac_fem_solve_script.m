% 1. Load mesh data (from heart-sa.zip)
load('heart-sa0.mat');  % Contains Omega_lin, Omega_quad structures

% 2. Build element basis objects
quad = 5;  % quadrature rule
Omega.e = fem_get_basis(Omega.p, quad, 'triangle');

% 3. Extract fiber directions and build D tensor per element
% D_e = (3/4)*f*f' + (1/4)*I for each element
D_e = 1;

% Mass matrix M: M_ij = ∫ φ_i φ_j dΩ
M = fem_assemble_block_matrix(@element_mass_matrix, Omega, u_space, u_space);

% Stiffness matrix K: K_ij = ∫ ∇φ_i · D ∇φ_j dΩ  
% (requires element-wise D tensor from fiber data)
K = fem_assemble_block_matrix(@element_stiffness_anisotropic, Omega, u_space, u_space);

% Initialize
U = zeros(n_nodes, 1);  % u = 0 everywhere initially
Z = zeros(n_nodes, 1);  % z = 0 everywhere initially

% Apply initial stimulation (Purkinje sites on Γ_2)
stim_nodes = find_stimulation_nodes(Omega, purkinje_sites, 7);  % 7mm radius
U(stim_nodes) = 1.0;  % Stimulate

% Time stepping
dt = 0.5;  % ms (adjust based on stability)
T_final = 500;  % ms (enough for full activation)
activation_time = inf(n_nodes, 1);

for n = 1:ceil(T_final/dt)
    t = n * dt;
    
    % Store old values
    U_old = U;
    Z_old = Z;
    
    % === Solve for U^{n+1} (Newton-Raphson) ===
    for iter = 1:max_iter
        vars.u.u = U;
        vars.z.u = Z_old;  % Use Z^n (semi-implicit)
        vars.u_old.u = U_old;
        vars.dt = dt;
        
        R_u = fem_assemble_block_residual(@u_equation_residual, Omega, u_space, vars);
        
        if norm(R_u) < tol
            break;
        end
        
        J_u = fem_assemble_block_matrix_perturbation(@u_equation_residual, ...
                                                     Omega, u_space, u_space, vars);
        
        dU = J_u \ R_u;
        U = U - dU;
    end
    
    % === Solve for Z^{n+1} (Linear solve) ===
    % (1 + ε*dt)*M*Z^{n+1} = M*Z^n - ε*k*dt*F2(U^n)
    vars.u_old.u = U_old;  % F2 uses U^n
    F2 = assemble_F2_vector(Omega, u_space, vars);
    Z = ((1 + eps*dt) * M) \ (M * Z_old - eps * k * dt * F2);
    
    % === Track activation times ===
    v = 125 * U - 80;  % Convert to mV
    newly_activated = (v > 20) & (activation_time == inf);
    activation_time(newly_activated) = t;
    
    % Check if fully activated
    if all(activation_time < inf)
        fprintf('Full activation at t = %.2f ms\n', t);
        break;
    end
end

T_total = max(activation_time);


lead_positions = [42, 165; 91, 179; 138, 148];  % Given in project
T_total_results = zeros(size(lead_positions, 1), 1);

for i = 1:size(lead_positions, 1)
    % Add CRT lead stimulation on Γ_3 within 7mm of lead position
    crt_nodes = find_boundary_nodes_near(Omega, lead_positions(i,:), 7, 3);  % Γ_3
    
    % Run simulation with both Purkinje + CRT stimulation
    [activation_time, T_total] = run_EP_simulation(Omega, purkinje_nodes, crt_nodes);
    
    T_total_results(i) = T_total;
    fprintf('Lead at (%.0f, %.0f): T_total = %.2f ms\n', ...
            lead_positions(i,1), lead_positions(i,2), T_total);
end

[T_opt, idx_opt] = min(T_total_results);
fprintf('Optimal lead position: (%.0f, %.0f) with T_total = %.2f ms\n', ...
        lead_positions(idx_opt,1), lead_positions(idx_opt,2), T_opt);