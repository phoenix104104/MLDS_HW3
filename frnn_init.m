function model = frnn_init(opts, class_map)
    
    fprintf('Initialize RNN...\n');

    model.opts = opts;
    
    model.M  = zeros(opts.hidden, 1, 'single'); % memory layer
    
    model.Wi = randn(opts.hidden, opts.num_dim, 'single') * 0.01;
    model.Bi = randn(opts.hidden, 1, 'single') * 0.01;
    model.Wm = randn(opts.hidden, opts.hidden, 'single') * 0.01;
    model.Bm = randn(opts.hidden, 1, 'single') * 0.01;
    model.Wo = randn(opts.num_label, opts.hidden, 'single') * 0.01;
    model.Bo = randn(opts.num_label, 1, 'single') * 0.01;
    model.Wc = randn(opts.num_class, opts.hidden, 'single') * 0.01;
    model.Bc = randn(opts.num_class, 1, 'single') * 0.01;
    
    % gradient buffer
    model.dWi = zeros(size(model.Wi), 'single');
    model.dBi = zeros(size(model.Bi), 'single');
    model.dWm = zeros(size(model.Wm), 'single');
    model.dBm = zeros(size(model.Bm), 'single');
    model.dWc = zeros(size(model.Wc), 'single');
    model.dBc = zeros(size(model.Bc), 'single');
    model.dWo = cell(opts.bptt_depth, 1);
    model.dBo = cell(opts.bptt_depth, 1);
    
    
    % for momentum, RMSProp
    model.mWi = zeros(size(model.Wi), 'single');
    model.mBi = zeros(size(model.Bi), 'single');
    model.mWm = zeros(size(model.Wm), 'single');
    model.mBm = zeros(size(model.Bm), 'single');
    model.mWo = zeros(size(model.Wo), 'single');
    model.mBo = zeros(size(model.Bo), 'single');
    model.mWc = zeros(size(model.Wc), 'single');
    model.mBc = zeros(size(model.Bc), 'single');

    % for RMSProp
    model.sWi = zeros(size(model.Wi), 'single');
    model.sBi = zeros(size(model.Bi), 'single');
    model.sWm = zeros(size(model.Wm), 'single');
    model.sBm = zeros(size(model.Bm), 'single');
    model.sWo = zeros(size(model.Wo), 'single');
    model.sBo = zeros(size(model.Bo), 'single');
    model.sWc = zeros(size(model.Wc), 'single');
    model.sBc = zeros(size(model.Bc), 'single');

    model.cost = zeros(opts.epoch, 1);
    model.class_map = class_map;
    model.class = cell(opts.num_class, 1);
    for c = 1:opts.num_class
        model.class{c} = find(model.class_map == c ); % index of y with the same class
    end
    
end
