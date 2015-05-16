function rnn_save_model(filename, model)
    
    filename = sprintf('%s.mat', filename);
    fprintf('Save %s\n', filename);
    save(filename, 'model');
    
end