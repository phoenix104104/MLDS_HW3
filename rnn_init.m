function model = rnn_init(opts)
    
    fprintf('Initialize RNN...\n');

    model.opts = opts;
    
    model.Wi = randn(opts.structure(2), opts.structure(1)) * 0.01;
    model.Bi = randn(opts.structure(2), 1) * 0.01;
    model.Wm = randn(opts.structure(2), opts.structure(2)) * 0.01;
    model.Bm = randn(opts.structure(2), 1) * 0.01;
    model.Wo = randn(opts.structure(3), opts.structure(2)) * 0.01;
    model.Bo = randn(opts.structure(3), 1) * 0.01;
    
    % for momentum
    model.mWi = zeros(size(model.Wi));
    model.mBi = zeros(size(model.Bi));
    model.mWm = zeros(size(model.Wm));
    model.mBm = zeros(size(model.Bm));
    model.mWo = zeros(size(model.Wo));
    model.mBo = zeros(size(model.Bo));
    
    model.M  = zeros(opts.hidden(1), 1); % memory layer
    
    % for training
    model.S = {}; % store memory layer
    model.z = {};
    model.a = {};
    model.delta = {};
    for i = 1:opts.bptt_depth
        model.S{i} = zeros(size(model.M));
        model.z{i} = zeros(size(model.M));
        model.a{i} = zeros(size(model.M));
        model.delta{i} = zeros(size(model.M));
    end
    model.delta{end+1} = zeros(size(model.M));
    
    
end
