function Me = mass_matrix(~, ~, teste, triale)
    ne = size(teste.y, 2);
    Me = zeros(ne);
    for i = 1:ne
        for j = 1:ne
            Me(i,j) = dot(teste.gw, teste.y(:,i).*triale.y(:,j));
        end
    end
end