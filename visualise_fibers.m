function visualise_fibers(Fibers, Omega)
% visualize_fibers plots the fiber vector field on top of the mesh
% Inputs:
%   Fibers: The fiber structure loaded from the .mat file
%   Omega:  The mesh structure (for context/plotting)

    figure;
    
    % 1. Plot the background mesh
    triplot(Omega.t(:,1:3), Omega.x(:,1), Omega.x(:,2),'Color', [0.8 0.8 0.8], 'LineWidth', 0.5);
    hold on;
    
    % 2. Extract Fiber Data
    n_nodes = size(Fibers.x, 1);
    n_rows  = size(Fibers.fib, 1);
    
    if n_rows == n_nodes
        % Nodal Data: Plot at node coordinates
        coords = Fibers.x;
        vecs   = Fibers.fib;
        title_str = sprintf('Fiber Orientation (Nodal, p=%d)', Fibers.p);
    else
        % Element Data: Plot at element centroids
        n_elems = size(Fibers.t, 1);
        coords = zeros(n_elems, 2);
        for i = 1:n_elems
            coords(i, :) = mean(Fibers.x(Fibers.t(i, 1:3), :), 1);
        end
        vecs = Fibers.fib;
        title_str = sprintf('Fiber Orientation (Element-wise, p=%d)', Fibers.p);
    end

    % 3. Normalize vectors for clean visualization
    vec_norms = sqrt(sum(vecs.^2, 2));
    vec_norms(vec_norms < 1e-6) = 1; % Avoid division by zero
    vecs_norm = vecs ./ vec_norms;
    
    % 4. Downsample if dense (plot every 5th arrow)
    step = 1;
    if size(coords, 1) > 1000, step = 1; 
    end
    
    % 5. Quiver Plot
    quiver(coords(1:step:end, 1), coords(1:step:end, 2),vecs_norm(1:step:end, 1), vecs_norm(1:step:end, 2),0.5, 'r', 'LineWidth', 1.5);
       
    title(title_str);
    xlabel('x (mm)'); ylabel('y (mm)');
    axis equal; grid on;
    legend('Mesh', 'Fiber Direction');
    hold off;
end