addpath('util');

% input_dir = '../data/easy2-c2-p1';
% train_dir = fullfile(input_dir, 'train');
% train_name = 'test2-c2-p1';
% train_name_list = 'train';

input_dir = '../feature_1_100_reduce/Vec';
train_dir = fullfile(input_dir, 'train');
data_list = 1;
train_name = 'train_001'; % define by yourself
train_name_list = {};
for i = 1:length(data_list)
    train_name_list{i} = sprintf('train_%03d.mat', data_list(i));
end

tic
%[Y_train, X_train] = rnn_load_data(train_dir, train_name_list);
[Y_train, X_train] = rnn_load_binary_data(train_dir, train_name_list);
toc


class_filename = 'class_map/class_mapping_100';
class_map = dlmread(class_filename);

fprintf('Mapping Y to C...\n');
Y_train = map_y_to_class(Y_train, class_map);

%opts.num_label      = find_max_label(Y_train);
%opts.num_label      = 3784;
opts.num_label      = 100;
opts.num_data       = length(Y_train);
opts.num_dim        = size(X_train{1}, 2);
opts.learning_rate  = 0.01;
opts.epoch          = 50;
opts.epoch_to_save  = 0;
opts.weight_decay   = 0.005;
opts.momentum       = 0.9;
opts.rmsprop_alpha  = 0.9;
opts.bptt_depth     = 5;
opts.gradient_thr   = 0.5;
opts.hidden         = 50;
opts.structure      = [opts.num_dim, opts.hidden, opts.num_label];
opts.activation     = 'sigmoid'; % options: sigmoid, relu
opts.update_grad    = 'sgd'; % options: sgd, rmsprop


parameter = sprintf('rnn_%s_hidden%d_lr%s_wd%s_m%s_bptt%s_thr%s', ...
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
for t = 1:N
    [Y_test, X_test] = rnn_load_data(test_dir, test_list{t}, 1);
    Y_test = map_y_to_class(Y_test, class_map);
    [res, y_pred, cost] = rnn_test(model, X_test, Y_test);
    result(t) = res;
    Y_pred{t} = y_pred;
end

filename = fullfile(opts.model_dir, sprintf('epoch%d.csv', opts.epoch));
save_kaggle_csv(filename, result);

%answer = dlmread(fullfile(input_dir, 'testing_ans'))+1;
answer = dlmread('google.ans');
acc = mean(answer == result)
