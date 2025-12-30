
%% Comparison Script for Thesis
% Runs simulations for all CRT configurations and saves results
% Configurations: Baseline (no CRT), CRT Site 1, CRT Site 2, CRT Site 3

%% Parameters
k = 8;
a = 0.15;
eps = 0.01;
D = 1;
dt = 0.1;
T_final = 500;
N_steps = ceil(T_final/dt);
stim_radius = 7;
purkinje_sites = [83, 75; 19, 44; 124, 10];
crt_sites = [42, 165; 91, 179; 138, 148];

% Configurations to run
config_names = {'Baseline (No CRT)', 'CRT Site 1', 'CRT Site 2', 'CRT Site 3'};
crt_options = [0, 1, 2, 3];  % 0 = no CRT, 1-3 = CRT sites

% Create results directory
results_dir = '../results';
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

%% Load mesh (once)
fprintf('Loading mesh and fiber data...\n');
mesh_data = load('heart-sa0.mat');
Omega = mesh_data.Omega;
Omega.e = fem_get_basis(Omega.p, 5, 'triangle');
Omega.name = 'Omega';
Omega.dm = 2;
n_nodes = size(Omega.x, 1);

% Load fiber directions
if isfield(mesh_data, 'Fibers')
    Fibers = mesh_data.Fibers;
    fprintf('✓ Loaded Fibers from mesh file\n');
else
    warning('Fibers not found in mesh file! Generating circumferential pattern...');
    Fibers = generate_fiber_field(Omega, 'circumferential');
end

%% Setup u space (once)
u_sp.name = 'u';
u_sp.dm = 1;
u_sp.p = Omega.p;
u_sp.x = Omega.x;
u_sp.t = Omega.t;
u_sp.e = Omega.e;

%% Assemble mass and stiffness matrices (once)
fprintf('Assembling mass and anisotropic stiffness matrices...\n');
M = fem_assemble_block_matrix(@mass_matrix, Omega, u_sp, u_sp);
K = fem_assemble_anisotropic_stiffness(Omega, u_sp, Fibers);
A = M/dt + K;
fprintf('  Assembly complete.\n\n');

%% Find Purkinje stimulation nodes (same for all)
bndry2 = unique(Omega.b(Omega.b(:,end)==2, 2:3));
bndry2 = bndry2(:);
stim_nodes_purkinje = [];
for i = 1:size(purkinje_sites,1)
    for j = 1:length(bndry2)
        if norm(Omega.x(bndry2(j),:) - purkinje_sites(i,:)) <= stim_radius
            stim_nodes_purkinje = [stim_nodes_purkinje; bndry2(j)];
        end
    end
end
stim_nodes_purkinje = unique(stim_nodes_purkinje);

%% Storage for results
results = struct();

%% Run simulations for each configuration
for config_idx = 1:length(crt_options)
    crt_option = crt_options(config_idx);
    config_name = config_names{config_idx};

    fprintf('========================================\n');
    fprintf('Running: %s\n', config_name);
    fprintf('========================================\n');

    %% Find CRT nodes for this configuration
    stim_nodes_crt = [];
    if crt_option > 0
        bndry3 = unique(Omega.b(Omega.b(:,end)==3, 2:3));
        bndry3 = bndry3(:);
        crt_site = crt_sites(crt_option, :);
        for j = 1:length(bndry3)
            if norm(Omega.x(bndry3(j),:) - crt_site) <= stim_radius
                stim_nodes_crt = [stim_nodes_crt; bndry3(j)];
            end
        end
        stim_nodes_crt = unique(stim_nodes_crt);
    end

    % Combine all stimulation nodes
    stim_nodes = unique([stim_nodes_purkinje; stim_nodes_crt]);
    fprintf('Purkinje: %d nodes, CRT: %d nodes, Total: %d nodes\n', ...
            length(stim_nodes_purkinje), length(stim_nodes_crt), length(stim_nodes));

    %% Initialize
    u = zeros(n_nodes, 1);
    z = zeros(n_nodes, 1);
    u(stim_nodes) = 1;
    activation_time = inf(n_nodes, 1);
    activation_time(stim_nodes) = 0;

    %% Time loop
    fully_activated = false;
    fprintf('Running time integration...\n');

    for n = 1:N_steps
        t = n*dt;

        % Ionic current
        f_ion = k*u.*(1-u).*(u-a) - u.*z;

        % Solve for u_new
        rhs = (M/dt)*u + M*f_ion;
        u_new = A \ rhs;

        % Update z
        g_z = -eps*(k*u.*(u-a-1) + z);
        z_new = z + dt*g_z;

        % Track activation
        newly_active = (u_new > 0.8) & (activation_time == inf);
        activation_time(newly_active) = t;

        % Check full activation
        if ~fully_activated && all(activation_time < inf)
            fprintf('  Full activation at t = %.1f ms\n', t);
            fully_activated = true;
        end

        % Update
        u = u_new;
        z = z_new;

        % Check stability
        if any(isnan(u)) || max(abs(u)) > 10
            error('Unstable at t=%.1f', t);
        end

        % Stop when propagation complete
        if fully_activated && max(u) < 0.1
            fprintf('  Propagation complete at t = %.1f ms\n', t);
            break;
        end
    end

    %% Compute statistics
    T_total = max(activation_time(activation_time < inf));
    T_mean = mean(activation_time(activation_time < inf));
    T_std = std(activation_time(activation_time < inf));

    fprintf('Statistics:\n');
    fprintf('  Total activation time: %.1f ms\n', T_total);
    fprintf('  Mean activation time: %.1f ms\n', T_mean);
    fprintf('  Std activation time: %.1f ms\n\n', T_std);

    %% Store results
    results(config_idx).name = config_name;
    results(config_idx).crt_option = crt_option;
    results(config_idx).activation_time = activation_time;
    results(config_idx).u_final = u;
    results(config_idx).z_final = z;
    results(config_idx).T_total = T_total;
    results(config_idx).T_mean = T_mean;
    results(config_idx).T_std = T_std;
    results(config_idx).stim_nodes = stim_nodes;
    results(config_idx).stim_nodes_crt = stim_nodes_crt;

    %% Save activation time map figure
    figure('Visible', 'off');
    act_plot = activation_time;
    act_plot(activation_time == inf) = NaN;
    trisurf(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), act_plot, ...
            'Facecolor', 'interp', 'LineStyle', 'none');
    colorbar;
    view(2);
    axis equal;
    colormap jet;
    title(sprintf('%s - Activation Time (T_{max} = %.1f ms)', config_name, T_total));
    xlabel('x (mm)');
    ylabel('y (mm)');

    % Save figure
    filename = sprintf('%s/activation_map_config_%d.png', results_dir, config_idx);
    saveas(gcf, filename);
    fprintf('Saved: %s\n', filename);
    close(gcf);
end

%% Save all results to .mat file
save(sprintf('%s/comparison_results.mat', results_dir), 'results', 'Omega', 'Fibers');
fprintf('\n✓ All results saved to: %s/comparison_results.mat\n', results_dir);

%% Create comparison plots
fprintf('\nGenerating comparison plots...\n');

%% 1. Side-by-side activation maps
figure('Position', [100, 100, 1600, 400]);
for config_idx = 1:4
    subplot(1, 4, config_idx);
    act_plot = results(config_idx).activation_time;
    act_plot(act_plot == inf) = NaN;
    trisurf(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), act_plot, ...
            'Facecolor', 'interp', 'LineStyle', 'none');
    colorbar;
    view(2);
    axis equal;
    colormap jet;
    caxis([0, max([results.T_total])]);  % Same color scale for all
    title(sprintf('%s\nT_{max} = %.1f ms', results(config_idx).name, results(config_idx).T_total));
    xlabel('x (mm)');
    ylabel('y (mm)');
end
sgtitle('Activation Time Comparison: All Configurations');
saveas(gcf, sprintf('%s/comparison_all_configs.png', results_dir));
saveas(gcf, sprintf('%s/comparison_all_configs.fig', results_dir));
fprintf('Saved: comparison_all_configs.png\n');

%% 2. Statistical comparison bar chart
figure('Position', [100, 100, 1000, 600]);

subplot(2,2,1);
bar([results.T_total]);
set(gca, 'XTickLabel', config_names, 'XTickLabelRotation', 45);
ylabel('Time (ms)');
title('Maximum Activation Time');
grid on;

subplot(2,2,2);
bar([results.T_mean]);
set(gca, 'XTickLabel', config_names, 'XTickLabelRotation', 45);
ylabel('Time (ms)');
title('Mean Activation Time');
grid on;

subplot(2,2,3);
bar([results.T_std]);
set(gca, 'XTickLabel', config_names, 'XTickLabelRotation', 45);
ylabel('Time (ms)');
title('Std Deviation of Activation Time');
grid on;

subplot(2,2,4);
% Activation time reduction compared to baseline
T_reduction = ([results.T_total] - results(1).T_total);
bar(T_reduction);
set(gca, 'XTickLabel', config_names, 'XTickLabelRotation', 45);
ylabel('Time reduction (ms)');
title('Activation Time Change vs Baseline');
grid on;
yline(0, 'r--', 'LineWidth', 1.5);

sgtitle('Statistical Comparison of CRT Configurations');
saveas(gcf, sprintf('%s/comparison_statistics.png', results_dir));
saveas(gcf, sprintf('%s/comparison_statistics.fig', results_dir));
fprintf('Saved: comparison_statistics.png\n');

%% 3. Activation time histograms
figure('Position', [100, 100, 1200, 800]);
for config_idx = 1:4
    subplot(2, 2, config_idx);
    act_times = results(config_idx).activation_time;
    act_times = act_times(act_times < inf);
    histogram(act_times, 50, 'FaceColor', 'b', 'EdgeColor', 'k');
    xlabel('Activation Time (ms)');
    ylabel('Number of Nodes');
    title(sprintf('%s', results(config_idx).name));
    grid on;
    xlim([0, max([results.T_total])]);
end
sgtitle('Activation Time Distribution');
saveas(gcf, sprintf('%s/comparison_histograms.png', results_dir));
saveas(gcf, sprintf('%s/comparison_histograms.fig', results_dir));
fprintf('Saved: comparison_histograms.png\n');

%% 4. Difference maps (CRT - Baseline)
figure('Position', [100, 100, 1400, 400]);
for config_idx = 2:4
    subplot(1, 3, config_idx-1);

    % Compute difference
    act_diff = results(config_idx).activation_time - results(1).activation_time;
    act_diff(results(1).activation_time == inf) = NaN;

    trisurf(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2), act_diff, ...
            'Facecolor', 'interp', 'LineStyle', 'none');
    colorbar;
    view(2);
    axis equal;
    colormap(redblue());  % Red = later, Blue = earlier
    title(sprintf('%s - Baseline\nΔT_{mean} = %.1f ms', ...
          results(config_idx).name, ...
          results(config_idx).T_mean - results(1).T_mean));
    xlabel('x (mm)');
    ylabel('y (mm)');
end
sgtitle('Activation Time Difference Maps (CRT - Baseline)');
saveas(gcf, sprintf('%s/comparison_difference_maps.png', results_dir));
saveas(gcf, sprintf('%s/comparison_difference_maps.fig', results_dir));
fprintf('Saved: comparison_difference_maps.png\n');

%% Generate summary table
fprintf('\n========================================\n');
fprintf('SUMMARY TABLE FOR THESIS\n');
fprintf('========================================\n');
fprintf('%-25s | T_max (ms) | T_mean (ms) | T_std (ms) | ΔT vs Baseline\n', 'Configuration');
fprintf('------------------------------------------------------------------------------------\n');
for config_idx = 1:4
    if config_idx == 1
        delta_str = '     -';
    else
        delta_T = results(config_idx).T_total - results(1).T_total;
        delta_str = sprintf('%+6.1f', delta_T);
    end
    fprintf('%-25s | %10.1f | %11.1f | %10.1f | %s\n', ...
            results(config_idx).name, ...
            results(config_idx).T_total, ...
            results(config_idx).T_mean, ...
            results(config_idx).T_std, ...
            delta_str);
end
fprintf('========================================\n\n');

%% Save summary to text file
fid = fopen(sprintf('%s/summary_table.txt', results_dir), 'w');
fprintf(fid, 'CARDIAC RESYNCHRONIZATION THERAPY (CRT) SIMULATION RESULTS\n');
fprintf(fid, '==========================================================\n\n');
fprintf(fid, 'Simulation Parameters:\n');
fprintf(fid, '  k = %.2f, a = %.3f, eps = %.3f\n', k, a, eps);
fprintf(fid, '  dt = %.2f ms, T_final = %.1f ms\n', dt, T_final);
fprintf(fid, '  Stimulation radius = %.1f mm\n', stim_radius);
fprintf(fid, '  Number of mesh nodes = %d\n', n_nodes);
fprintf(fid, '  Anisotropic conductivity: σ_parallel/σ_perp = 4.0\n\n');
fprintf(fid, 'Results Summary:\n');
fprintf(fid, '%-25s | T_max (ms) | T_mean (ms) | T_std (ms) | ΔT vs Baseline\n', 'Configuration');
fprintf(fid, '------------------------------------------------------------------------------------\n');
for config_idx = 1:4
    if config_idx == 1
        delta_str = '     -';
    else
        delta_T = results(config_idx).T_total - results(1).T_total;
        delta_str = sprintf('%+6.1f', delta_T);
    end
    fprintf(fid, '%-25s | %10.1f | %11.1f | %10.1f | %s\n', ...
            results(config_idx).name, ...
            results(config_idx).T_total, ...
            results(config_idx).T_mean, ...
            results(config_idx).T_std, ...
            delta_str);
end
fclose(fid);
fprintf('✓ Summary table saved to: %s/summary_table.txt\n\n', results_dir);

fprintf('========================================\n');
fprintf('ALL SIMULATIONS COMPLETE!\n');
fprintf('========================================\n');
fprintf('Results saved in: %s/\n', results_dir);
fprintf('  - comparison_results.mat (all data)\n');
fprintf('  - activation_map_config_*.png (individual maps)\n');
fprintf('  - comparison_all_configs.png (side-by-side)\n');
fprintf('  - comparison_statistics.png (bar charts)\n');
fprintf('  - comparison_histograms.png (distributions)\n');
fprintf('  - comparison_difference_maps.png (CRT effects)\n');
fprintf('  - summary_table.txt (text results)\n');
fprintf('========================================\n');

%% Helper function for red-blue colormap
function cmap = redblue()
    % Creates a diverging red-white-blue colormap
    n = 64;
    r = [linspace(0, 1, n/2), ones(1, n/2)];
    g = [linspace(0, 1, n/2), linspace(1, 0, n/2)];
    b = [ones(1, n/2), linspace(1, 0, n/2)];
    cmap = [r', g', b'];
end
