addpath('util');

input_dir = '../feature_1_100_reduce/Vec';
train_dir = fullfile(input_dir, 'train');
data_list = 1:73;
train_name = 'train_all'; % define by yourself
train_name_list = {};
for i = 1:length(data_list)
    train_name_list{i} = sprintf('train_%03d.mat', data_list(i));
end

class_filename = 'class_map/class_mapping_100';
class_map = dlmread(class_filename);

opts.num_label      = 3784;
opts.num_dim        = 100;
opts.num_class      = max(class_map);
opts.learning_rate  = 0.01;
opts.epoch          = 100;
opts.epoch_to_save  = 1;
opts.weight_decay   = 0.05;
opts.momentum       = 0.9;
opts.rmsprop_alpha  = 0.9;
opts.bptt_depth     = 3;
opts.gradient_thr   = 0.5;
opts.hidden         = 50;
opts.structure      = [opts.num_dim, opts.hidden, opts.num_label+opts.num_class];
opts.activation     = 'sigmoid'; % options: sigmoid, relu
opts.update_grad    = 'sgd'; % options: sgd, rmsprop


parameter = sprintf('frnn_%s_hidden%d_lr%s_wd%s_m%s_bptt%s_thr%s', ...
                 train_name, opts.hidden, ...
                 num2str(opts.learning_rate), num2str(opts.weight_decay), ...
                 num2str(opts.momentum), num2str(opts.bptt_depth), ...
                 num2str(opts.gradient_thr));
             
opts.model_dir = fullfile('../model', parameter);
if( ~exist(opts.model_dir, 'dir') )
    fprintf('mkdir %s\n', opts.model_dir);
    mkdir(opts.model_dir);
end

             
model = frnn_init(opts, class_map);
model = frnn_train_all(model, train_dir, train_name_list);


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
fprintf('FRNN testing...\n');
for t = 1:N
    [Y_test, X_test] = rnn_load_data(test_dir, test_list{t}, 1);
    [res, y_pred, cost] = frnn_test(model, X_test, Y_test);
    result(t) = res;
    Y_pred{t} = y_pred;
end

filename = fullfile(opts.model_dir, sprintf('epoch%d.csv', opts.epoch));
save_kaggle_csv(filename, result);

% answer = dlmread(fullfile(input_dir, 'testing_ans'));
answer = dlmread('google.ans');
acc = mean(answer == result)
