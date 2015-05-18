function save_binary_data(filename, dim)

    fprintf('Load %s\n', filename);
    %D = dlmread(filename);
    
    fmt = ['%d ', repmat(['%f '], 1, dim)];
    f = fopen(filename);
    C = textscan(f, fmt);
    fclose(f);
    
    n = size(C{1}, 1);
    D = zeros(n, dim+1);
    for i = 1:dim+1
        D(:, i) = C{i};
    end
    
    filename = sprintf('%s.mat', filename);
    fprintf('Save %s\n', filename);
    save(filename, 'D');

end