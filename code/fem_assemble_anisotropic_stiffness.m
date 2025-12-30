function [K] = fem_assemble_anisotropic_stiffness(Omega, u_sp, Fibers)
    % FEM_ASSEMBLE_ANISOTROPIC_STIFFNESS - Assemble global stiffness matrix
    % with anisotropic conductivity tensor based on fiber directions
    %
    % Inputs:
    %   Omega   - Domain structure (mesh)
    %   u_sp    - Solution space structure
    %   Fibers  - Fiber structure containing fiber directions (Fibers.fib)
    %
    % Output:
    %   K       - Global stiffness matrix with anisotropic conductivity

    % Error checking
    fem_check_space(Omega, Omega.dm);
    fem_check_space(u_sp, Omega.dm);

    if size(Omega.t,1) ~= size(u_sp.t,1)
        error('fem: the number of elements in the domain inconsistent with solution space');
    end

    % Check if Fibers structure is valid
    if ~isfield(Fibers, 'fib')
        error('Fibers structure must contain field "fib" with fiber directions');
    end

    % Calculate global matrix size and element matrix size
    n_dof = u_sp.dm * size(u_sp.x, 1);  % total degrees of freedom
    ne = u_sp.dm * size(u_sp.t, 2);     % DOFs per element

    % Initialize sparse global matrix
    K = sparse(n_dof, n_dof);
    Ke = zeros(ne, ne);

    % Loop over all elements
    n_elems = size(Omega.t, 1);
    for e = 1:n_elems
        % Map basis function quantities to the element
        [metrics] = fem_compute_metrics(Omega, e);
        [teste] = map_element_quantities(metrics, u_sp.e);
        [triale] = map_element_quantities(metrics, u_sp.e);

        % Compute element stiffness matrix with anisotropic conductivity
        % Pass element index e, not connectivity row
        Ke = stiffness_matrix_anisotropic(Omega, e, teste, triale, Fibers);

        % Add element matrix contributions to global matrix K
        for i = 1:u_sp.dm
            for j = 1:u_sp.dm
                vi = u_sp.dm * (u_sp.t(e,:) - 1) + i;
                vj = u_sp.dm * (u_sp.t(e,:) - 1) + j;
                vei = 1:size(u_sp.t, 2);
                vei = u_sp.dm * (vei' - 1) + i;
                vej = 1:size(u_sp.t, 2);
                vej = u_sp.dm * (vej' - 1) + j;
                K(vi, vj) = K(vi, vj) + Ke(vei, vej);
            end
        end
    end
end

% Helper functions (copied from fem_assemble_block_matrix.m)
function [metrics] = fem_compute_metrics(Omega, elem)
    % Extract local weights for element mapping
    x = Omega.x(Omega.t(elem,:), 1);
    y = Omega.x(Omega.t(elem,:), 2);

    % Compute spatial coordinates at Gauss points
    metrics.x = Omega.e.y(:,:) * x;
    metrics.y = Omega.e.y(:,:) * y;

    % Compute derivatives of spatial coordinate map at Gauss points
    dxdxi1 = Omega.e.dy(:,:,1) * x;
    dxdxi2 = Omega.e.dy(:,:,2) * x;
    dydxi1 = Omega.e.dy(:,:,1) * y;
    dydxi2 = Omega.e.dy(:,:,2) * y;

    % Compute Jacobian of the mapping
    metrics.j = abs(dxdxi1 .* dydxi2 - dxdxi2 .* dydxi1);

    % Compute transform matrix for gradient
    ja = dxdxi1 .* dydxi2 - dxdxi2 .* dydxi1;
    metrics.m11 =  dydxi2 ./ ja;
    metrics.m12 = -dxdxi2 ./ ja;
    metrics.m21 = -dydxi1 ./ ja;
    metrics.m22 =  dxdxi1 ./ ja;
end

function [teste] = map_element_quantities(metrics, e)
    teste = e;
    teste.gw = e.gw .* metrics.j;

    for i = 1:size(e.y, 2)
        teste.dy(:,i,1) = metrics.m11 .* e.dy(:,i,1) + metrics.m21 .* e.dy(:,i,2);
        teste.dy(:,i,2) = metrics.m12 .* e.dy(:,i,1) + metrics.m22 .* e.dy(:,i,2);
    end
end
