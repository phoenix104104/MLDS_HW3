function model = rnn_init(opts)
    
    fprintf('Initialize RNN...\n');

    model.opts = opts;
    
    model.M  = zeros(opts.hidden(1), 1, 'single'); % memory layer
    
    model.Wi = (randn(opts.structure(2), opts.structure(1), 'single') * 0.01);
    model.Bi = (randn(opts.structure(2), 1, 'single') * 0.01);
    model.Wm = (randn(opts.structure(2), opts.structure(2), 'single') * 0.01);
    model.Bm = (randn(opts.structure(2), 1, 'single') * 0.01);
    model.Wo = (randn(opts.structure(3), opts.structure(2), 'single') * 0.01);
    model.Bo = (randn(opts.structure(3), 1, 'single') * 0.01);
    
    % gradient buffer
    model.dWi = (zeros(size(model.Wi), 'single'));
    model.dBi = (zeros(size(model.Bi), 'single'));
    model.dWm = (zeros(size(model.Wm), 'single'));
    model.dBm = (zeros(size(model.Bm), 'single'));
    model.dWo = (zeros(size(model.Wo), 'single'));
    model.dBo = (zeros(size(model.Bo), 'single'));
    
    % for momentum
    model.mWi = (zeros(size(model.Wi), 'single'));
    model.mBi = (zeros(size(model.Bi), 'single'));
    model.mWm = (zeros(size(model.Wm), 'single'));
    model.mBm = (zeros(size(model.Bm), 'single'));
    model.mWo = (zeros(size(model.Wo), 'single'));
    model.mBo = (zeros(size(model.Bo), 'single'));
    
    model.cost = zeros(opts.epoch, 1);
    
end
