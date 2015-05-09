function net = dnn_init(opts)

    net.opts = opts;
    net.W = {};
    net.B = {};
    for i = 1:length(opts.structure)-1
        net.W{end+1} = randn(opts.structure(i+1), opts.structure(i));
        net.B{end+1} = randn(opts.structure(i+1), 1);
    end
end