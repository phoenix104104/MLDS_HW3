addpath('util');

input_dir = '../data/test2-c2-p1';
train_name = 'train';
train_filename = fullfile(input_dir, 'train', sprintf('%s', train_name));
normalize = 0;


tic
[Y_train, X_train, mu, sigma] = rnn_load_data(train_filename, normalize);
%[Y_train, X_train, mu, sigma] = rnn_load_binary_data(train_filename, normalize);
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
opts.epoch          = 1;
opts.weight_decay   = 0;
opts.momentum       = 0.0;
%opts.rmsprop_alpha  = 0.9;
opts.bptt_depth     = 4;
opts.gradient_thr   = 0.5;
opts.hidden         = 100;
opts.structure      = [num_dim, opts.hidden, num_class];
opts.activation     = 'sigmoid'; % options: sigmoid, relu
opts.update_grad    = 'sgd';


model = rnn_init(opts);
model = rnn_train(model, X_train, Y_train);


parameter = sprintf('%s_hidden%d_epoch%d_lr%s_wd%s_m%s_bptt%s_thr%s.rnn', ...
                 train_name, opts.hidden, opts.epoch, ...
                 num2str(opts.learning_rate), num2str(opts.weight_decay), ...
                 num2str(opts.momentum), num2str(opts.bptt_depth), ...
                 num2str(opts.gradient_thr));
             
% model_filename = sprintf('../model/%s.model', parameter);
% rnn_save_model(model_filename, model);


test_dir = fullfile(input_dir, 'test');
fprintf('Load test data from %s\n', test_dir);
L = dir(test_dir);
test_list = {};
for i = 1:length(L)
    if(L(i).name(1) == '.')
        continue;
    end
    test_list{end+1} = L(i).name;
end

N = length(test_list);
result = zeros(N, 1);
Y_pred = cell(N, 1);
fprintf('RNN testing...\n');
for t = 1:N
    test_filename = fullfile(test_dir, test_list{t});
    [Y_test, X_test, mu, sigma] = rnn_load_data(test_filename, normalize, mu, sigma, 1);
    [res, y_pred, cost] = rnn_test(model, X_test, Y_test);
    result(t) = res-1;
    Y_pred{t} = y_pred;
end

% filename = sprintf('../pred/%s.csv', parameter);
% save_kaggle_csv(filename, result);
answer = dlmread(fullfile(input_dir, 'testing_ans'));
acc = mean(answer == result)
