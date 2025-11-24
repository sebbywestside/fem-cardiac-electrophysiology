function Ce = element_matrix_c(testsp, trialsp, teste, triale)

% This function computes the element matrix for the 2D discrete Laplacian
%   c(B, v) = ∫_Ω ∇B · ∇v dΩ
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
%        Be - the element matrix
%
% by David Nordsletten
% Jan 2015

  % Getting local row / local column sizes ...
  ne=size(testsp.t,2);
  me=size(trialsp.t,2);

  % initialising element matrix
  Ce = zeros(testsp.dm * ne, trialsp.dm * me);

% adding ocnstants from question
D0 = 1;

  for rw = 1:ne
      for col = 1:me          
          
          % diffusion term
          grad_term = dot(teste.gw, teste.dy(:, rw, 1) .* triale.dy(:, col, 1) + teste.dy(:, rw, 2) .* triale.dy(:, col, 2));
          
          % element matrix
          Ce(rw,col) = D0 * grad_term;
      end
  end
end

