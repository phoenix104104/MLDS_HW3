addpath('util');

input_dir = '../test1-c3';
train_filename = fullfile(input_dir, 'train', 'train');
feature_name = '';

normalize = 0;

tic
[Y_train, X_train, mu, sigma] = rnn_load_data(train_filename, normalize);
toc

num_class = 4;
num_data = length(Y_train);
num_dim = size(X_train{1}, 2);
fprintf('Input data size = %d\n', num_data);
fprintf('Feature dimension = %d\n', num_dim);

opts.num_class      = num_class;
opts.num_data       = num_data;
opts.num_dim        = num_dim;
opts.normalize      = normalize;
opts.learning_rate  = 0.01;
opts.epoch          = 10;
opts.weight_decay   = 0;
opts.momentum       = 0.9;
%opts.rmsprop_alpha  = 0.9;
opts.dropout_prob   = 0.0;
opts.bptt_depth     = 3;
opts.gradient_thr   = 0.5;
opts.hidden         = [10];
opts.structure      = [num_dim, opts.hidden, num_class];
opts.activation     = 'sigmoid'; % options: sigmoid, relu
opts.update_grad    = 'sgd';


model = rnn_init(opts);
model = rnn_train(model, X_train, Y_train);

N = 100;
result = zeros(N, 1);
Y_pred = cell(N, 1);
for t = 1:N
    test_filename = fullfile(input_dir, 'test', sprintf('test_%02d', t-1));
    [Y_test, X_test, mu, sigma] = rnn_load_data(test_filename, normalize, mu, sigma);
    [res, y_pred, cost] = rnn_test(model, X_test, Y_test);
    result(t) = res-1;
    Y_pred{t} = y_pred;
end

answer = dlmread(fullfile(input_dir, 'testing_ans'));
acc = mean(answer == result)


%model_filename = '../model/test.rnn';
%rnn_save_model(model_filename, model);
