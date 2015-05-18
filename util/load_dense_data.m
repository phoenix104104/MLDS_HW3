function [y, X] = load_dense_data(filename, silence)
    
    if( ~exist('silence', 'var') )
        silence = 0;
    end
    if( ~silence )
        fprintf('Load %s\n', filename); 
    end
    X = dlmread(filename);
    y = int32(X(:, 1)) + 1;
    X = X(:, 2:end);
end