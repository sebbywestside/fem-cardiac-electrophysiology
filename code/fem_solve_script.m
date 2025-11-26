%  AFEM solve -- (FEM Tutorials)
%
%    This function assembles an FEM problem and solves it.  
%
%  by David Nordsletten
%  Jan 2015
%
clear

  quad=5;
  etype = 'quadrilateral';

  NoElems1D=10;
  h=[1 1] / NoElems1D;
  nonlinear_tol = 1e-6; % setting the nonlinear tolerance
  iter = 0; % initializing the interation counter
  residual_vector_norm = 1; % initializing the residual vector norm

% Making Domain function space ...
  Omega.name = 'space'; % field name ...
  Omega.dm = 2; % two dimensional domain ...
  Omega.p  = 1; % linear polynomials ...
  Omega.e=fem_get_basis(Omega.p, quad, etype);
  [Omega.x, Omega.t, Omega.b] = fem_make_mesh_2D(NoElems1D, h, Omega.e); % make the mesh ...

% Making variable function space ...
  vel.name = 'vel'; % field name ...
  vel.dm = 2; % vector field ...
  vel.p  = 2; % quadratic polynomials ...
  vel.e=fem_get_basis(vel.p, quad, etype); 
  [vel.x, vel.t, vel.b] = fem_make_mesh_2D(NoElems1D, h, vel.e); 
% Initializing the coefficients for the velocity field to zero (no nodes x number of unknowns per node)
  vel.u = zeros(vel.dm * size(vel.x,1), 1);
  n_vel = size(vel.u,1); 

% Making variable function space ...
  pres.name = 'pres'; % field name ...
  pres.dm = 1; % scalar field ...
  pres.p  = 1; % quadratic polynomials ...
  pres.e=fem_get_basis(pres.p, quad, etype); 
  [pres.x, pres.t, pres.b] = fem_make_mesh_2D(NoElems1D, h, pres.e); 
% Initializing the coefficients for the pressure field to zero (no nodes x number of unknowns per node)
  pres.u = zeros(size(pres.x,1),1); 
  n_pres = size(pres.u,1);

while (iter == 0) || (residual_vector_norm > nonlinear_tol)
  % Making a FEM object storing my unknown variables (must do this every iteration) ...
    vars.vel = vel;
    vars.pres = pres;

  % Making the residual vector and applying boundary conditions (velocity) ...
    R1 = fem_assemble_block_residual(@darcy_momentum_eqn, Omega, vel, vars);
    for i = 1:size(vel.x,1)
      if(vel.b(i) == 0) 
        continue
      end

      nn = vel.dm * (i - 1);
      R1(nn+1:nn+2) = -[ 1 - vel.u(nn+1); 0 - vel.u(nn+2)];
    end
  % Making the residual vector and applying boundary conditions (on species B) ...
    R2 = fem_assemble_block_residual(@darcy_mass_eqn, Omega, pres, vars);
    R2(1) = 0;

  % Making the global residual vector + computing the norm ...
    R = [R1; R2];
    residual_vector_norm = norm(R,2); 
    )])
    if(residual_vector_norm < nonlinear_tol)
      continue
    end

  % Creating the discrete matrix operators corresponding to operators a, b, and c
    disp(['        constructing the Jacobian blocks ...'])
    A  = fem_assemble_block_matrix_perturbation(@darcy_momentum_eqn, Omega, vel, vel, vars); 
    B  = fem_assemble_block_matrix_perturbation(@darcy_momentum_eqn, Omega, vel, pres, vars); 
    C  = fem_assemble_block_matrix_perturbation(@darcy_mass_eqn, Omega, pres, vel, vars); 
    D = sparse(size(pres.u,1),size(pres.u,1));

  % Editing block matrices for dirichlet conditions ...
    for i = 1:size(vel.x,1)
      if(vel.b(i) == 0) 
        continue
      end

      nn = vel.dm * (i - 1);
      A(nn+1:nn+2,:) = 0;
      A(nn+1:nn+2, nn+1:nn+2) = eye(2);
      B(nn+1:nn+2,:) = 0;
    end
    C(1,:) = 0; D(1,1) = 1;

  % Composing the Jacobian from our Jacobian blocks ...
    disp(['        assembly of the Global Jacobian ...'])
    J = [ A B; C D];

  % Apply Newton Raphson ...
    disp(['        solving for the NR update ...'])
    U = J \ R;
    vel.u(1:n_vel) = vel.u(1:n_vel) - U(1:n_vel);
    pres.u(1:n_pres) = pres.u(1:n_pres) - U(n_vel+1:end);

  % Update the iteration counter ...
    iter = iter + 1;

    disp(['  ']) % skip a line so output is prettier
end

  % Plotting the velocity field ...
  quiver(vel.x(:,1), vel.x(:,2), vel.u(1:vel.dm:n_vel), vel.u(2:vel.dm:n_vel))

  % plotting the pressure field ...
  figure(2); trisurf(pres.t(:,1:3), pres.x(:,1), pres.x(:,2), pres.u,'Facecolor','interp','LineStyle','none')
  hold; trisurf(pres.t(:,2:4), pres.x(:,1), pres.x(:,2), pres.u,'Facecolor','interp','LineStyle','none')

