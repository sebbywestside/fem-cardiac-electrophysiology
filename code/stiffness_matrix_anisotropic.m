function Ke = stiffness_matrix_anisotropic(Omega, elem_idx, teste, triale, Fibers)
    % STIFFNESS_MATRIX_ANISOTROPIC - Compute element stiffness matrix with
    % anisotropic conductivity tensor D = (3/4)*f*f' + (1/4)*I
    %
    % Inputs:
    %   Omega     - Domain structure
    %   elem_idx  - Element index (which element we're assembling)
    %   teste     - Test function structure with gradients
    %   triale    - Trial function structure with gradients
    %   Fibers    - Fiber structure containing fiber directions
    %
    % Output:
    %   Ke        - Element stiffness matrix

    ne = size(teste.y, 2);  % number of basis functions
    nq = length(teste.gw);  % number of quadrature points
    Ke = zeros(ne);

    % Get fiber direction for this element
    % Fibers.fib contains fiber directions (n_elems x 2 or n_nodes x 2)

    if size(Fibers.fib, 1) == size(Omega.t, 1)
        % Element-wise fiber data
        f = Fibers.fib(elem_idx, :)';  % Column vector [f1; f2]
    else
        % Nodal fiber data - average over element nodes
        elem_nodes = Omega.t(elem_idx, :);  % all nodes in this element
        f = mean(Fibers.fib(elem_nodes, :), 1)';  % Column vector
    end

    % Normalize fiber direction (should already be normalized, but ensure it)
    f_norm = norm(f);
    if f_norm > 1e-10
        f = f / f_norm;
    else
        f = [1; 0];  % default direction if fiber is zero
    end

    % Construct conductivity tensor D = (3/4)*f*f' + (1/4)*I
    % This gives faster conductivity along fiber direction
    D = (3/4) * (f * f') + (1/4) * eye(2);

    % D is a 2x2 matrix:
    % D = [D11  D12]
    %     [D21  D22]
    D11 = D(1,1);
    D12 = D(1,2);
    D21 = D(2,1);
    D22 = D(2,2);

    % Assemble element stiffness matrix
    % K_ij = ∫∫ ∇φ_i · D · ∇φ_j dΩ
    %      = ∫∫ [∂φ_i/∂x1, ∂φ_i/∂x2] · D · [∂φ_j/∂x1; ∂φ_j/∂x2] dΩ
    %
    % Expanding:
    % = ∫∫ ( D11 * ∂φ_i/∂x1 * ∂φ_j/∂x1 + D12 * ∂φ_i/∂x1 * ∂φ_j/∂x2
    %      + D21 * ∂φ_i/∂x2 * ∂φ_j/∂x1 + D22 * ∂φ_i/∂x2 * ∂φ_j/∂x2 ) dΩ

    for i = 1:ne
        for j = 1:ne
            % teste.dy(:,i,1) = ∂φ_i/∂x1 at all quadrature points
            % teste.dy(:,i,2) = ∂φ_i/∂x2 at all quadrature points
            % teste.gw = quadrature weights

            integrand = D11 * teste.dy(:,i,1) .* triale.dy(:,j,1) + D12 * teste.dy(:,i,1) .* triale.dy(:,j,2) + D21 * teste.dy(:,i,2) .* triale.dy(:,j,1) + D22 * teste.dy(:,i,2) .* triale.dy(:,j,2);

            Ke(i,j) = dot(teste.gw, integrand);
        end
    end
end
