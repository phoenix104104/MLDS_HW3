function model = rnn_load_model(filename)

    filename = sprintf('%s.mat', filename);
    fprintf('Load %s\n', filename);
    M = load(filename);
    model = M.model;
    
end