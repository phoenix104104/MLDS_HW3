function save_dense_data(filename, y, X)

    X = [double(y), X];
    fprintf('Save %s\n', filename);
    dlmwrite(filename, X, ' ');

end
