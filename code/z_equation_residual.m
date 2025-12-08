function [Re] = z_equation_residual(e, testsp, teste, vars)
    % This forms: (1 + ε*dt)*M*Z^{n+1} - M*Z^n + ε*k*dt*F2(U^n) = 0
    
    z_local = vars.z.u(vars.z.t(e,:));
    z_old   = vars.z_old.u(vars.z_old.t(e,:));
    u_old   = vars.u_old.u(vars.u_old.t(e,:));  
    z_gp = teste.y * z_local;
    z_old_gp = teste.y * z_old;
    u_old_gp = teste.y * u_old;
    
    k = 8; a = 0.15; eps = 0.01; dt = vars.dt;
    
    % F2 term: u(u - a - 1)
    F2_gp = u_old_gp .* (u_old_gp - a - 1);
    
    ne = size(testsp.t, 2);
    Re = zeros(ne, 1);
    
    for i = 1:ne
        Re(i) = (1 + eps*dt) * dot(teste.gw, teste.y(:,i) .* z_gp) ...
              - dot(teste.gw, teste.y(:,i) .* z_old_gp) ...
              + eps * k * dt * dot(teste.gw, teste.y(:,i) .* F2_gp);
    end
end