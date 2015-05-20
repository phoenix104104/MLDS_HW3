function y_max = find_max_label(Y)
    
    N = length(Y);
    m = zeros(N, 1);
    for i = 1:N
        i
        max(Y{i})
        m(i) = max(Y{i});
    end
    y_max = max(m);
    
end