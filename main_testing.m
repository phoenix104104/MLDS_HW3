addpath('util');

input_dir = '../feature_1_100_reduce/Vec';
train_name = 'train_1to2';

opts.learning_rate  = 0.01;
opts.epoch_to_save  = 3;
opts.weight_decay   = 0;
opts.momentum       = 0.0;
%opts.rmsprop_alpha  = 0.9;
opts.bptt_depth     = 3;
opts.gradient_thr   = 0.5;
opts.hidden         = 1000;
opts.activation     = 'sigmoid'; % options: sigmoid, relu
opts.update_grad    = 'sgd';


parameter = sprintf('%s_hidden%d_lr%s_wd%s_m%s_bptt%s_thr%s', ...
                 train_name, opts.hidden, ...
                 num2str(opts.learning_rate), num2str(opts.weight_decay), ...
                 num2str(opts.momentum), num2str(opts.bptt_depth), ...
                 num2str(opts.gradient_thr));
             
opts.model_dir = fullfile('../model', parameter);

epoch = 50;
model_filename = fullfile(opts.model_dir, sprintf('epoch%d.rnn', iter));
model = rnn_load_model(model_filename);
             

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
    test_filename{1} = test_list{t};
    [Y_test, X_test] = rnn_load_data(test_dir, test_filename, 1);
    [res, y_pred, cost] = rnn_test(model, X_test, Y_test);
    result(t) = res;
    Y_pred{t} = y_pred;
end

filename = fullfile(opts.model_dir, sprintf('epoch%d.csv', epoch));
save_kaggle_csv(filename, result);

% answer = dlmread(fullfile(input_dir, 'testing_ans'));
answer = dlmread('google.ans');
acc = mean(answer == result)
