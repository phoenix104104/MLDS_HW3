function model = dnn_init(opts)
    
    fprintf('Initialize DNN...\n');

    model.opts = opts;
    model.W = {};
    model.B = {};
    model.mW = {}; % momentum for W
    model.mB = {}; % momentum for B
    for i = 1:length(opts.structure)-1
        model.W{i}  = randn(opts.structure(i+1), opts.structure(i)) * 0.01;
        model.B{i}  = randn(opts.structure(i+1), 1) * 0.01;
        model.mW{i} = zeros(size(model.W{i}));
        model.mB{i} = zeros(size(model.B{i}));
    end
    
end