function model = rnn_clear_memory(model)

    model.M  = zeros(size(model.M));
    
    % for training
    for i = 1:model.opts.bptt_depth
        model.S{i} = zeros(size(model.M));
        model.z{i} = zeros(size(model.M));
        model.a{i} = zeros(size(model.M));
        model.delta{i} = zeros(size(model.M));
    end
    model.delta{model.opts.bptt_depth+1} = zeros(size(model.M));
    
        
end