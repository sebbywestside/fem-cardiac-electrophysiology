%% TEST_ANISOTROPIC_CONDUCTIVITY
% Quick test to verify the anisotropic conductivity tensor implementation
%
% This script tests:
% 1. Loading fiber directions
% 2. Assembling anisotropic stiffness matrix
% 3. Comparing with isotropic case
%
% by Sebastian (AFEM Project)

clear; close all; clc;

fprintf('=== Testing Anisotropic Conductivity Tensor Implementation ===\n\n');

%% Load mesh
fprintf('1. Loading mesh...\n');
mesh_data = load('heart-sa/heart-sa0.mat');
fprintf('   Variables in file: %s\n', strjoin(fieldnames(mesh_data), ', '));

Omega = mesh_data.Omega;
Omega.e = fem_get_basis(Omega.p, 5, 'triangle');
Omega.name = 'Omega';
Omega.dm = 2;
fprintf('   Mesh loaded: %d nodes, %d elements\n', size(Omega.x,1), size(Omega.t,1));

%% Load or generate fiber field
fprintf('\n2. Loading fiber field from file...\n');
if isfield(mesh_data, 'Fibers')
    Fibers = mesh_data.Fibers;
    fprintf('   ✓ Loaded Fibers from mesh file\n');
    fprintf('   Fiber data size: %d x %d\n', size(Fibers.fib));
else
    fprintf('   ⚠ Fibers not in file, generating circumferential pattern...\n');
    Fibers = generate_fiber_field(Omega, 'circumferential');
    fprintf('   Generated fiber field: %d fiber vectors\n', size(Fibers.fib, 1));
end

%% Setup solution space
fprintf('\n3. Setting up solution space...\n');
u_sp.name = 'u';
u_sp.dm = 1;
u_sp.p = Omega.p;
u_sp.x = Omega.x;
u_sp.t = Omega.t;
u_sp.e = Omega.e;
fprintf('   Solution space configured\n');

%% Assemble isotropic stiffness matrix (for comparison)
fprintf('\n4. Assembling ISOTROPIC stiffness matrix...\n');
tic;
K_iso = fem_assemble_block_matrix(@stiffness_matrix, Omega, u_sp, u_sp);
t_iso = toc;
fprintf('   Assembly time: %.3f seconds\n', t_iso);
fprintf('   Matrix size: %d x %d\n', size(K_iso));
fprintf('   Non-zeros: %d (%.2f%% sparse)\n', nnz(K_iso), 100*nnz(K_iso)/numel(K_iso));

%% Assemble anisotropic stiffness matrix
fprintf('\n5. Assembling ANISOTROPIC stiffness matrix with fibers...\n');
tic;
K_aniso = fem_assemble_anisotropic_stiffness(Omega, u_sp, Fibers);
t_aniso = toc;
fprintf('   Assembly time: %.3f seconds\n', t_aniso);
fprintf('   Matrix size: %d x %d\n', size(K_aniso));
fprintf('   Non-zeros: %d (%.2f%% sparse)\n', nnz(K_aniso), 100*nnz(K_aniso)/numel(K_aniso));

%% Compare matrices
fprintf('\n6. Comparing isotropic vs anisotropic matrices...\n');
K_diff = K_aniso - K_iso;
fprintf('   Norm of difference: %.4e\n', norm(K_diff, 'fro'));
fprintf('   Relative difference: %.2f%%\n', 100*norm(K_diff,'fro')/norm(K_iso,'fro'));
fprintf('   Max absolute difference: %.4e\n', full(max(max(abs(K_diff)))));

%% Visualize fiber field
fprintf('\n7. Visualizing fiber field...\n');
figure('Name', 'Fiber Field Visualization');
visualise_fibers(Fibers, Omega);

%% Test conductivity tensor at a few sample points
fprintf('\n8. Testing conductivity tensor construction...\n');
test_fibers = [1, 0; 0, 1; 1/sqrt(2), 1/sqrt(2)];  % horizontal, vertical, diagonal
test_names = {'Horizontal', 'Vertical', 'Diagonal'};

for i = 1:size(test_fibers, 1)
    f = test_fibers(i,:)';
    D = (3/4) * (f * f') + (1/4) * eye(2);

    fprintf('   %s fiber [%.2f, %.2f]:\n', test_names{i}, f(1), f(2));
    fprintf('     D = [%.3f  %.3f]\n', D(1,1), D(1,2));
    fprintf('         [%.3f  %.3f]\n', D(2,1), D(2,2));
    fprintf('     Eigenvalues: [%.3f, %.3f]\n', eig(D)');
    fprintf('     (Should be ~1.0 along fiber, ~0.25 perpendicular)\n');
end

fprintf('\n=== All tests completed successfully! ===\n');
fprintf('\nNext steps:\n');
fprintf('  - Run fem_solve_script.m to see the full cardiac EP simulation\n');
fprintf('  - Compare activation times with/without fiber directions\n');
