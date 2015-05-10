function [y, X] = dnn_load_data(filename)
    
    fprintf('Load %s\n', filename);
    X = dlmread(filename);
    y = int32(X(:, 1)) + 1;
    X = X(:, 2:end);
end