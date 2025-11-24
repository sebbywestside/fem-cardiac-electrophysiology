function Be = element_matrix_b(testsp, trialsp, teste, triale)

% This function computes the element matrix for the 2D discrete Laplacian
%   b(A, v) = ∫_Ω 4KA v dx
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
  Be = zeros(testsp.dm * ne, trialsp.dm * me);

% adding ocnstants from question
K=1; ratio = 4;

  for rw = 1:ne
      for col = 1:me          
          
          cx = - ratio * K * dot(teste.gw, teste.y(:, rw) .* triale.y(:, col));
          
          Be(rw,col) = cx;
      end
  end
end

