function Ae = element_matrix_a(testsp, trialsp, teste, triale)

% This function computes the element matrix for the 2D discrete Laplacian
% a(A, w) = ∫_Ω (∇A · ∇w + Aw) dΩ
% Inputs --
%
%        testsp             - the test space to use
%
%        trialsp            - the trial space to use
%
%        teste   - the element / basis object for test functions ...
%
%        triale  - the element / basis object for trial functions ...
%
%
% Outputs --
%        Ae - the element matrix
%
% by David Nordsletten
% Jan 2015

  % Getting local row / local column sizes ...
  ne=size(testsp.t,2);
  me=size(trialsp.t,2);

  % initialising element matrix
  Ae = zeros(testsp.dm * ne, trialsp.dm * me);

% adding ocnstants from question
D=1;K=1;

  for rw = 1:ne
      for col = 1:me
          % diffusion term
          ax = dot(teste.gw , teste.dy(:,rw,1).*teste.dy(:,col,1) + teste.dy(:,rw,2).*teste.dy(:,col,2) );
          % reaction term
          cx = K * dot(teste.gw, teste.y(:, rw) .* teste.y(:, col));
          % operator A
          Ae(rw,col) = ax + cx;
      end
  end
end

