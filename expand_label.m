function Y = expand_label(y, n_class)
    
    n = length(y);
    Y = zeros(n, n_class);
    for i = 1:n
        Y(i, y(i)) = 1;
    end
    
end