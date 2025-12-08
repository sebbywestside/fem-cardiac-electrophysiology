function [Re] = u_equation_residual(e, testsp, teste, vars)
% cardiac_u_eqn -- Cardiac Electrophysiology: u-equation residual
%
% Computes element residual for the membrane potential equation:
%
%   R_i = (1/dt) ∫ φ_i (u^{n+1} - u^n) dΩ 
%       + ∫ ∇φ_i · D∇u^{n+1} dΩ 
%       - ∫ φ_i f(u^{n+1}, z^n) dΩ = 0
%
% where f(u,z) = ku(1-u)(u-a) - uz  (Aliev-Panfilov model)
%
% Using ISOTROPIC D = 1 for validation (ignore fibers for now)
%
% Inputs:
%   e       - element index
%   testsp  - test function space
%   teste   - test functions evaluated at quadrature points (mapped)
%   vars    - structure containing:
%             vars.u      - current solution with .u coefficients
%                           also contains .dt, .k, .a parameters
%             vars.u_old  - previous time step solution
%             vars.z      - gating variable (from previous time step)
%
% Output:
%   Re      - element residual vector

    % Parameters 
    k = vars.u.k;
    a = vars.u.a;
    D = 1.0;  
    
    % time step
    dt = vars.u.dt;
   
    % Current u^{n+1} iterate
    u_local = vars.u.u(vars.u.t(e,:));
    
    % Previous time step u^n
    u_old_local = vars.u_old.u(vars.u_old.t(e,:));
    
    % Gating variable z^n (semi-implicit: use from previous time step)
    z_local = vars.z.u(vars.z.t(e,:));
    
    % Evaluate fields at quadrature points using mapped basis
    % u^{n+1} at quadrature points
    u_qp = vars.ue.y * u_local;
    
    % u^n at quadrature points
    u_old_qp = vars.u_olde.y * u_old_local;
    
    % z^n at quadrature points
    z_qp = vars.ze.y * z_local;
    
    % Gradient of u^{n+1} at quadrature points
    grad_u_qp(:,1) = vars.ue.dy(:,:,1) * u_local;  
    grad_u_qp(:,2) = vars.ue.dy(:,:,2) * u_local;  
    
    % Compute ionic current f(u,z) = ku(1-u)(u-a) - uz
    f_ion_qp = k * u_qp .* (1 - u_qp) .* (u_qp - a) - u_qp .* z_qp;
    
    % Assemble element residual

    % Number of local nodes
    ne = size(testsp.t, 2);
    Re = zeros(testsp.dm * ne, 1);
    
    for i = 1:ne
        % Term 1: Mass term 
        mass_term = (1/dt) * dot(teste.gw, teste.y(:,i) .* (u_qp - u_old_qp));
        
        % Term 2: Stiffness term 
        stiff_term = D * dot(teste.gw, teste.dy(:,i,1) .* grad_u_qp(:,1) + teste.dy(:,i,2) .* grad_u_qp(:,2));
        
        % Term 3: Ionic term =
        ion_term = -dot(teste.gw, teste.y(:,i) .* f_ion_qp);
        
        % Total residual for node i
        Re(i) = mass_term + stiff_term + ion_term;
    end

end