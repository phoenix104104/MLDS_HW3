function [Y, X] = dnn_load_data(filename, n_class)
    
    fprintf('Load %s\n', filename);
    X = dlmread(filename);
    y = int32(X(:, 1)) + 1;
    n = length(y);
    Y = zeros(n, n_class);
    for i = 1:n
        Y(i, y(i)) = 1;
    end
    
    X = X(:, 2:end);
end