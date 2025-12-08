% initialising constants 
k = 8.0;
a = 0.15;
epsilon = 0.01;

% Time stepping
dt = 0.5;           
T_final = 400;     
N_steps = ceil(T_final / dt);

% NR$ parameters
newton_tol = 1e-6;
newton_max_iter = 20;

% srtimulation params
stim_radius = 7.0;  
u_stim = 1.0;      

% Purkinje sites 
purkinje_sites = [83, 75;    % Site 1
                  19, 44;    % Site 2
                  124, 10];  % Site 3
% CRT sites 
crt_sites = [42, 165;   % Option 1
             91, 179;   % Option 2
             138, 148]; % Option 3

% Load the heart mesh
mesh_file = 'heart-sa0.mat';

fprintf('\nMesh info:\n');
fprintf('  Nodes: %d\n', size(Omega.x, 1));
fprintf('  Elements: %d\n', size(Omega.t, 1));
fprintf('  Polynomial order: %d\n', Omega.p);

quad = 5;  % Lyness quadrature rule
Omega.e = fem_get_basis(Omega.p, quad, 'triangle');
fprintf('  Built basis object (quad rule = %d)\n', quad);
Omega.name = 'Omega';
Omega.dm = 2;

n_nodes = size(Omega.x, 1);
n_elems = size(Omega.t, 1);

patch_indices = Omega.b(:, end);
for p_idx = unique(patch_indices)'
    count = sum(patch_indices == p_idx);
    fprintf('  Patch %d: %d boundary faces\n', p_idx, count);
end
fprintf('  Patch 2 = Γ₂ (Purkinje sites)\n');
fprintf('  Patch 3 = Γ₃ (CRT lead sites)\n');

% u (membrane potential) 
u.name = 'u';
u.dm = 1;
u.p = Omega.p;
u.x = Omega.x;
u.t = Omega.t;
u.e = Omega.e;
u.b = Omega.b;

% z (gating variable) 
z.name = 'z';
z.dm = 1;
z.p = Omega.p;
z.x = Omega.x;
z.t = Omega.t;
z.e = Omega.e;
z.b = Omega.b;

% initialise solutions
u.u = zeros(n_nodes, 1);       % u^{n+1} current iterate
u_old = u; u_old.name = 'u_old';
u_old.u = zeros(n_nodes, 1);   % u^n previous time step

z.u = zeros(n_nodes, 1);       % z^{n+1} current
z_old = z; z_old.name = 'z_old';
z_old.u = zeros(n_nodes, 1);   % z^n previous time step

% Activation time tracking
activation_time = inf(n_nodes, 1);
activation_threshold = 0.8;  

% Extract unique boundary nodes for each patch
fprintf('\nIdentifying boundary nodes...\n');

% Boundary array: [element, node1, node2, patch_index]
% For linear elements (p=1), nodes are in columns 2 and 3
bndry_nodes_patch2 = unique(Omega.b(Omega.b(:,end) == 2, 2:3));
bndry_nodes_patch3 = unique(Omega.b(Omega.b(:,end) == 3, 2:3));
bndry_nodes_patch2 = bndry_nodes_patch2(:);
bndry_nodes_patch3 = bndry_nodes_patch3(:);

fprintf('  Γ₂ boundary nodes: %d\n', length(bndry_nodes_patch2));
fprintf('  Γ₃ boundary nodes: %d\n', length(bndry_nodes_patch3));

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

stim_nodes = stim_nodes_purkinje;

% visualising mesh 
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

figure(1); clf;
triplot(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), 'k-', 'LineWidth', 0.3);
hold on;

% Plot boundary nodes
plot(Omega.x(bndry_nodes_patch2, 1), Omega.x(bndry_nodes_patch2, 2),'b.', 'MarkerSize', 8);
plot(Omega.x(bndry_nodes_patch3, 1), Omega.x(bndry_nodes_patch3, 2),'g.', 'MarkerSize', 8);

% Plot stimulation nodes
plot(Omega.x(stim_nodes, 1), Omega.x(stim_nodes, 2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');

% Plot Purkinje and CRT sites
plot(purkinje_sites(:,1), purkinje_sites(:,2),'b^', 'MarkerSize', 12, 'MarkerFaceColor', 'b', 'LineWidth', 2);
plot(crt_sites(:,1), crt_sites(:,2),'g', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'LineWidth', 2);

title('Heart Mesh with Stimulation Sites');
xlabel('x_1 (mm)'); ylabel('x_2 (mm)');
axis equal; grid on;
drawnow;

%
u.u(stim_nodes) = u_stim;
u_old.u(stim_nodes) = u_stim;
activation_time(stim_nodes) = 0;
fprintf('\nApplied initial stimulation: u = %.2f at %d nodes\n', u_stim, length(stim_nodes));

for n = 1:N_steps
    t_current = n * dt;
    
    %Step 1: Solve for u^{n+1} using Newton-Raphson
    % Initial guess: u^{n+1} = u^n
    u.u = u_old.u;
    
    for iter = 1:newton_max_iter
       
        u.dt = dt;
        u.k = k;
        u.a = a;
        vars.u = u;
        vars.u_old = u_old;
        vars.z = z_old; 
        
        % Assemble residual using course function
        R_u = fem_assemble_block_residual(@u_equation_residual, Omega, u, vars);

        R_u(stim_nodes) = 0;
        
        % Compute residual norm
        res_norm = norm(R_u, 2);
        
        % Debug output for first time step
        if n == 1 && iter <= 5
            fprintf('  Newton iter %d: res = %.4e, max|R| = %.4e, u:[%.4f,%.4f], du_norm = ', ...
                    iter, res_norm, max(abs(R_u)), min(u.u), max(u.u));
        end
        
        if res_norm < newton_tol
            if n == 1, fprintf('converged\n'); end
            break;
        end
        
        % Assemble Jacobian using perturbation method
        J_u = fem_assemble_block_matrix_perturbation(@u_equation_residual, Omega, u, u, vars);
        
        % Solve for Newton update
        du = J_u \ (-R_u);
        
        % Debug: print du norm
        if n == 1 && iter <= 5
            fprintf('%.4e\n', norm(du));
        end
        
        % Keep stimulated nodes fixed (Dirichlet BC)
        du(stim_nodes) = 0;
        
        % Update solution
        u.u = u.u + du;
        
        % DON'T clamp during Newton iteration - breaks convergence
        % Will clamp after Newton converges
    end
    
    if iter == newton_max_iter && res_norm > newton_tol
        warning('Newton did not converge at t=%.2f ms (res=%.2e)', t_current, res_norm);
    end
    
    % Clamp u to [0, 1] AFTER Newton convergence for physical validity
    u.u = max(0, min(1, u.u));
    
    % Step 2: Update z^{n+1} (explicit formula)
    % From: (1 + ε*dt)*z^{n+1} = z^n - ε*k*dt*u^n*(u^n - a - 1)
    u_n = u_old.u;
    F2_nodal = u_n .* (u_n - a - 1);
    z.u = (z_old.u - epsilon * k * dt * F2_nodal) / (1 + epsilon * dt);
    
    % === Step 3: Track activation times ===
    newly_activated = (u.u > activation_threshold) & (activation_time == inf);
    activation_time(newly_activated) = t_current;
    n_activated = sum(activation_time < inf);
    
    % === Step 4: Update for next time step ===
    u_old.u = u.u;
    z_old.u = z.u;
    
    %% === Progress output ===
    if mod(n, 10) == 0 || n == 1
        fprintf('t = %6.1f ms | iter = %2d | res = %.2e | activated: %d/%d (%.1f%%)\n', ...
                t_current, iter, res_norm, n_activated, n_nodes, 100*n_activated/n_nodes);
    end
    
    %% === Plot progress ===
    if mod(n, plot_interval) == 0
        figure(2); clf;
        trisurf(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), u.u, ...
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
end