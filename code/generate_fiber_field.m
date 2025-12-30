function Fibers = generate_fiber_field(Omega, fiber_type)
    % GENERATE_FIBER_FIELD - Generate fiber direction field for cardiac tissue
    %
    % Inputs:
    %   Omega       - Mesh structure with nodes (Omega.x) and elements (Omega.t)
    %   fiber_type  - Type of fiber field to generate:
    %                 'circumferential' - fibers tangent to circles around heart center
    %                 'radial' - fibers pointing radially from heart center
    %                 'uniform' - uniform fiber direction
    %
    % Output:
    %   Fibers      - Fiber structure with fields:
    %                 .fib  - Fiber directions (n_nodes x 2 or n_elems x 2)
    %                 .x    - Node coordinates
    %                 .t    - Element connectivity
    %                 .p    - Polynomial order

    if nargin < 2
        fiber_type = 'circumferential';
    end

    % Copy mesh data
    Fibers.x = Omega.x;
    Fibers.t = Omega.t;
    Fibers.p = Omega.p;

    % Find geometric center of the domain
    x_center = mean(Omega.x(:,1));
    y_center = mean(Omega.x(:,2));

    % For more accurate center, use the RV/LV boundary center
    % Here we'll use a weighted center based on the inner boundaries
    if isfield(Omega, 'b')
        % Try to find inner boundary (patch 1 is usually outer)
        inner_boundary_nodes = unique(Omega.b(Omega.b(:,end) ~= 1, 2:3));
        if ~isempty(inner_boundary_nodes)
            inner_boundary_nodes = inner_boundary_nodes(:);
            x_center = mean(Omega.x(inner_boundary_nodes, 1));
            y_center = mean(Omega.x(inner_boundary_nodes, 2));
        end
    end

    fprintf('Heart center for fiber generation: (%.2f, %.2f)\n', x_center, y_center);

    % Generate fiber directions at nodes
    n_nodes = size(Omega.x, 1);
    fiber_directions = zeros(n_nodes, 2);

    for i = 1:n_nodes
        % Vector from center to node
        dx = Omega.x(i,1) - x_center;
        dy = Omega.x(i,2) - y_center;
        r = sqrt(dx^2 + dy^2);

        if r < 1e-10
            % At center, use default direction
            fiber_directions(i,:) = [1, 0];
        else
            switch fiber_type
                case 'circumferential'
                    % Tangent to circle (perpendicular to radial)
                    % Radial direction: [dx/r, dy/r]
                    % Tangent (CCW): [-dy/r, dx/r]
                    fiber_directions(i,:) = [-dy/r, dx/r];

                case 'radial'
                    % Radial direction (pointing outward)
                    fiber_directions(i,:) = [dx/r, dy/r];

                case 'uniform'
                    % Uniform horizontal direction
                    fiber_directions(i,:) = [1, 0];

                otherwise
                    error('Unknown fiber_type: %s', fiber_type);
            end
        end

        % Normalize (should already be normalized, but ensure it)
        norm_val = norm(fiber_directions(i,:));
        if norm_val > 1e-10
            fiber_directions(i,:) = fiber_directions(i,:) / norm_val;
        end
    end

    Fibers.fib = fiber_directions;

    fprintf('Generated %s fiber field with %d nodal fiber directions\n', ...
            fiber_type, n_nodes);
end
