addpath('util');

%input_dir = '../data/test2-c3';
%train_name_list{1} = fullfile(input_dir, 'train', 'train');

input_dir = '../feature_1_100/Vec';
data_list = 1:2;
train_name = 'train_1to2'; % define by yourself
train_name_list = {};
for i = 1:length(data_list)
    train_name_list{i} = fullfile(input_dir, 'train', sprintf('train_%03d.mat', data_list(i)));
end

tic
%[Y_train, X_train] = rnn_load_data(train_name_list);
[Y_train, X_train] = rnn_load_binary_data(train_name_list);
toc


num_class = find_max_label(Y_train);
num_data = length(Y_train);
num_dim = size(X_train{1}, 2);
fprintf('Input data size = %d\n', num_data);
fprintf('Feature dimension = %d\n', num_dim);

opts.num_class      = num_class;
opts.num_data       = num_data;
opts.num_dim        = num_dim;
opts.learning_rate  = 0.01;
opts.epoch          = 1000;
opts.epoch_to_save  = 10;
opts.weight_decay   = 0.0005;
opts.momentum       = 0.9;
%opts.rmsprop_alpha  = 0.9;
opts.bptt_depth     = 3;
opts.gradient_thr   = 0.5;
opts.hidden         = 1000;
opts.structure      = [num_dim, opts.hidden, num_class];
opts.activation     = 'sigmoid'; % options: sigmoid, relu
opts.update_grad    = 'sgd';


parameter = sprintf('%s_hidden%d_lr%s_wd%s_m%s_bptt%s_thr%s', ...
                 train_name, opts.hidden, ...
                 num2str(opts.learning_rate), num2str(opts.weight_decay), ...
                 num2str(opts.momentum), num2str(opts.bptt_depth), ...
                 num2str(opts.gradient_thr));
             
opts.model_dir = fullfile('../model', parameter);
if( ~exist(opts.model_dir, 'dir') )
    fprintf('mkdir %s\n', opts.model_dir);
    mkdir(opts.model_dir);
end

             
model = rnn_init(opts);
model = rnn_train(model, X_train, Y_train);


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
test_filename = {};
for t = 1:N
    test_filename{1} = fullfile(test_dir, test_list{t});
    [Y_test, X_test] = rnn_load_data(test_filename, 1);
    [res, y_pred, cost] = rnn_test(model, X_test, Y_test);
    result(t) = res-1;
    Y_pred{t} = y_pred;
end

filename = sprintf('../pred/%s.csv', parameter);
save_kaggle_csv(filename, result);
% answer = dlmread(fullfile(input_dir, 'testing_ans'));
% acc = mean(answer == result)
