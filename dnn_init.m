function model = dnn_init(opts)
    
    fprintf('Initialize DNN...\n');

    model.opts = opts;
    model.W = {};
    model.B = {};
    for i = 1:length(opts.structure)-1
        model.W{end+1} = randn(opts.structure(i+1), opts.structure(i)) * 0.01;
        model.B{end+1} = randn(opts.structure(i+1), 1) * 0.01;
    end
end