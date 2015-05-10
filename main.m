data_dir = '../data';
% 
% feature_name = 'mnist-train.dense.raw';
% filename = fullfile(data_dir, feature_name);
% [y_train, X_train] = dnn_load_data(filename);
% 
% feature_name = 'mnist-test.dense.raw';
% filename = fullfile(data_dir, feature_name);
% [y_valid, X_valid] = dnn_load_data(filename);
% 
% 
% fprintf('Data normalization...\n');
% [X_train, mu, sigma] = normalize_data(X_train);
% [X_valid, mu, sigma] = normalize_data(X_valid, mu, sigma);

num_class   = length(unique(y_train));
num_data    = size(y_train, 1);
num_dim     = size(X_train, 2);

fprintf('Input data size = %d\n', num_data);
fprintf('Feature dimension = %d\n', num_dim);

opts.num_class      = num_class;
opts.learning_rate  = 0.05;
opts.batch_size     = 256;
opts.epoch          = 100;
opts.weight_decay   = 0.005;
opts.momentum       = 0.9;
opts.dropout_prob   = 0.5;
opts.hidden         = [256];
opts.structure      = [num_dim, opts.hidden, num_class];
opts.activation     = 'ReLU';

model = dnn_init(opts);

model = dnn_train(model, X_train, y_train, X_valid, y_valid);


