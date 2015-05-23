function C = map_y_to_class(Y, class_list)
    
    n_seq = length(Y);
    C = cell(n_seq, 1);
    for i = 1:n_seq
        C{i} = class_list(Y{i});
    end

end