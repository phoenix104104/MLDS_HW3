addpath('util');

train_data_dir = '../toy/train';
feature_name = '';

tic
[Y_train, X_train] = rnn_load_data(train_data_dir, feature_name);
%[y_train, X_train, y_valid, X_valid] ...
%   = load_hw2_data(train_filename, valid_filename, train_list_name, valid_list_name);
toc

num_data = length(Y_train);
num_dim = size(X_train{1}, 2);
fprintf('Input data size = %d\n', num_data);
fprintf('Feature dimension = %d\n', num_dim);

opts.learning_rate  = 0.01;
opts.epoch          = 50;
opts.weight_decay   = 0;
opts.momentum       = 0.9;
opts.rmsprop_alpha  = 0.9;
opts.dropout_prob   = 0.0;
opts.bptt_depth     = 3;
opts.gradient_thr   = 0.5; %%
opts.hidden         = [4];
opts.structure      = [num_dim, opts.hidden, num_dim];
opts.activation     = 'sigmoid';
opts.update_grad    = 'sgd';


model = rnn_init(opts);
model = rnn_train(model, X_train, Y_train);

result = zeros(10, 1);
Y_pred = cell(10, 1);
for t = 1:10
    test_data_dir = sprintf('../toy/test%03d', t-1);
    [Y_test, X_test] = rnn_load_data(test_data_dir, feature_name);
    [res, y_pred, cost] = rnn_test(model, X_test, Y_test);
    result(t) = res-1;
    Y_pred{t} = y_pred;
end

answer = dlmread('../toy/testing_ans');
[answer, result]
