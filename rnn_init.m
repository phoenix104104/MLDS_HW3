function model = rnn_init(opts)
    
    fprintf('Initialize RNN...\n');

    model.opts = opts;
    
    model.M  = zeros(opts.hidden(1), 1); % memory layer
    
    model.Wi = randn(opts.structure(2), opts.structure(1)) * 0.01;
    model.Bi = randn(opts.structure(2), 1) * 0.01;
    model.Wm = randn(opts.structure(2), opts.structure(2)) * 0.01;
    model.Bm = randn(opts.structure(2), 1) * 0.01;
    model.Wo = randn(opts.structure(3), opts.structure(2)) * 0.01;
    model.Bo = randn(opts.structure(3), 1) * 0.01;
    
    % gradient buffer
    model.dWi = zeros(size(model.Wi));
    model.dBi = zeros(size(model.Bi));
    model.dWm = zeros(size(model.Wm));
    model.dBm = zeros(size(model.Bm));
    model.dWo = zeros(size(model.Wo));
    model.dBo = zeros(size(model.Bo));
    
    % for momentum
    model.mWi = zeros(size(model.Wi));
    model.mBi = zeros(size(model.Bi));
    model.mWm = zeros(size(model.Wm));
    model.mBm = zeros(size(model.Bm));
    model.mWo = zeros(size(model.Wo));
    model.mBo = zeros(size(model.Bo));
    
    model.cost = zeros(opts.epoch, 1);
    
end
