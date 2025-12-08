function Ke = stiffness_matrix(~, ~, teste, triale)
    ne = size(teste.y, 2);
    Ke = zeros(ne);
    for i = 1:ne
        for j = 1:ne
            Ke(i,j) = dot(teste.gw, teste.dy(:,i,1).*triale.dy(:,j,1) + teste.dy(:,i,2).*triale.dy(:,j,2));
        end
    end
end