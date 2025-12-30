

%% Parameters
k = 8;
a = 0.15;
eps = 0.01;
%D = 1;
dt = 0.1;
T_final = 500;
N_steps = ceil(T_final/dt);
stim_radius = 8.5;
purkinje_sites = [83, 75; 19, 44; 124, 10];
crt_sites = [42, 165; 91, 179; 138, 148];
crt_option = 2;  % choose which CRT lead position (1, 2, or 3) 0 is all 3 in a subplot (not yet implemented)
%crt_option_array = [1, 2, 3];

%% Load mesh
mesh_data = load('heart-sa3.mat');
fprintf('Variables in mesh file: %s\n', strjoin(fieldnames(mesh_data), ', '));

% Extract Omega
Omega = mesh_data.Omega;
Omega.e = fem_get_basis(Omega.p, 5, 'triangle');
Omega.name = 'Omega';
Omega.dm = 2;
n_nodes = size(Omega.x, 1);

%% Load fiber directions from mesh file
if isfield(mesh_data, 'Fibers')
    Fibers = mesh_data.Fibers;
    fprintf('✓ Loaded Fibers from mesh file\n');
    fprintf('  Fiber data size: %d x %d\n', size(Fibers.fib));
    fprintf('  Fiber fields: %s\n', strjoin(fieldnames(Fibers), ', '));
else
    warning('Fibers not found in mesh file! Generating circumferential pattern...');
    Fibers = generate_fiber_field(Omega, 'circumferential');
end

%% Setup u space
u_sp.name = 'u';
u_sp.dm = 1;
u_sp.p = Omega.p;
u_sp.x = Omega.x;
u_sp.t = Omega.t;
u_sp.e = Omega.e;

%% Assemble mass and stiffness matrices
M = fem_assemble_block_matrix(@mass_matrix, Omega, u_sp, u_sp);

% Assemble ANISOTROPIC stiffness matrix with fiber-based conductivity tensor
% D = (3/4)*f*f' + (1/4)*I  (incorporated inside the assembly)
fprintf('Assembling anisotropic stiffness matrix with fiber directions...\n');
K = fem_assemble_anisotropic_stiffness(Omega, u_sp, Fibers);
fprintf('  Stiffness matrix assembled: %d x %d\n', size(K));

% Note: D parameter is now incorporated into the conductivity tensor
% No longer multiply by scalar D
A = M/dt + K;

%% Find which boundary each Purkinje site is on
for i = 1:size(purkinje_sites,1)
    for patch = 1:max(Omega.b(:,end))
        bndry_nodes = unique(Omega.b(Omega.b(:,end)==patch, 2:3));
        bndry_nodes = bndry_nodes(:);
        
        min_dist = 40000;
        for j = 1:length(bndry_nodes)
            dist = norm(Omega.x(bndry_nodes(j),:) - purkinje_sites(i,:));
            if dist < min_dist
                min_dist = dist;
            end
        end
        
        if min_dist <= stim_radius
            fprintf('Purkinje site %d is on boundary %d (min_dist = %.2f)\n', i, patch, min_dist);
        end
    end
end

% Get all boundary nodes (or specific boundaries)
all_bndry = unique(Omega.b(:, 2:3));
all_bndry = all_bndry(:);
% Find Purkinje stimulation nodes (ONLY on boundary 2)
bndry2 = unique(Omega.b(Omega.b(:,end)==2, 2:3));
bndry2 = bndry2(:);

stim_nodes_purkinje = [];
for i = 1:size(purkinje_sites,1)
    %for j = 1:length(bndry2)
    for j = 1:length(all_bndry)   
        %if norm(Omega.x(bndry2(j),:) - purkinje_sites(i,:)) <= stim_radius
         %    stim_nodes_purkinje = [stim_nodes_purkinje; bndry2(j)];
        %end
        if norm(Omega.x(all_bndry(j),:) - purkinje_sites(i,:)) <= stim_radius
            stim_nodes_purkinje = [stim_nodes_purkinje; all_bndry(j)];
        end
    end
end
stim_nodes_purkinje = unique(stim_nodes_purkinje);


% CRT nodes (boundary patch 3)
bndry3 = unique(Omega.b(Omega.b(:,end)==3, 2:3));
bndry3 = bndry3(:);
stim_nodes_crt = [];
crt_site = crt_sites(crt_option, :);
for j = 1:length(bndry3)
    if norm(Omega.x(bndry3(j),:) - crt_site) <= stim_radius
        stim_nodes_crt = [stim_nodes_crt; bndry3(j)];
    end
end
stim_nodes_crt = unique(stim_nodes_crt);

% combine all stimulation nodes
stim_nodes = unique([stim_nodes_purkinje; stim_nodes_crt]);
fprintf('Purkinje stim nodes: %d, CRT stim nodes: %d, Total: %d\n',length(stim_nodes_purkinje), length(stim_nodes_crt), length(stim_nodes));


%% Initialize
u = zeros(n_nodes, 1);
z = zeros(n_nodes, 1);
u(stim_nodes) = 1;
activation_time = inf(n_nodes, 1);
activation_time(stim_nodes) = 0;

% set up live plot
figure;
h = trisurf(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), u, 'Facecolor', 'interp', 'LineStyle', 'none');
%h = trisurf(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), u); % see mesh

colorbar; caxis([0 1]); view(2); axis equal;
colormap;
title('t = 0 ms');
drawnow;

%% Time loop
fully_activated = false;

for n = 1:N_steps
    t = n*dt;
    
    % ionic current at old time
    f_ion = k*u.*(1-u).*(u-a) - u.*z;
    
    % solve for u_new
    rhs = (M/dt)*u + M*f_ion;
    u_new = A \ rhs;
    
    % update z (forward Euler)
    g_z = -eps*(k*u.*(u-a-1) + z);
    z_new = z + dt*g_z;
    
    % track activation
    newly_active = (u_new > 0.8) & (activation_time == inf);
    activation_time(newly_active) = t;
    
    % check if all nodes have activated
    if ~fully_activated && all(activation_time < inf)
        fprintf('Full activation at t = %.1f ms\n', t);
        fully_activated = true;
    end
    
    % update
    u = u_new;
    z = z_new;
    
    
    % update plot every step
    set(h, 'CData', u);
    title(sprintf('t = %.0f ms', t));
    drawnow;
    
    % check stability
    if any(isnan(u)) || max(abs(u)) > 10
        error('Unstable at t=%.1f', t);
    end
    
    % stop when propagation has ended (all repolarized after full activation)
    if fully_activated && max(u) < 0.01
        fprintf('Propagation complete at t = %.1f ms\n', t);
        break;
    end
end

%% Results
T_total = max(activation_time(activation_time < inf));
fprintf('Total activation time: %.1f ms\n', T_total);

% plot activation time map
figure;
act_plot = activation_time;
act_plot(activation_time == inf) = NaN;
trisurf(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), act_plot, 'Facecolor', 'interp', 'LineStyle', 'none');
colorbar; view(2); axis equal;
colormap;
title(sprintf('Activation Time (T = %.1f ms)', T_total));

%% Visualize fiber directions
fprintf('\nVisualizing fiber directions...\n');
figure;
visualise_fibers(Fibers, Omega);



