addpath('util');

input_dir = '../feature_15_100/Vec';
train_filename = fullfile(input_dir, 'train', 'train_001.mat');
normalize = 0;



tic
%[Y_train, X_train, mu, sigma] = rnn_load_data(train_filename, normalize);
[Y_train, X_train, mu, sigma] = rnn_load_binary_data(train_filename, normalize);
toc


num_class = 24803;
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
opts.hidden         = [1000];
opts.structure      = [num_dim, opts.hidden, num_class];
opts.activation     = 'sigmoid'; % options: sigmoid, relu
opts.update_grad    = 'sgd';


model = rnn_init(opts);
model = rnn_train(model, X_train, Y_train);

%{
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
%m = zeros(N, 1);
for t = 1:N
    test_filename = fullfile(test_dir, test_list{t});
    [Y_test, X_test, mu, sigma] = rnn_load_data(test_filename, normalize, mu, sigma, 1);
%     for j = 1:length(Y_test)
%         m(end+1) = max(Y_test{j});
%     end
    %[res, y_pred, cost] = rnn_test(model, X_test, Y_test);
    %result(t) = res-1;
    %Y_pred{t} = y_pred;
end
max(m)
%}
% 
% answer = dlmread(fullfile(input_dir, 'testing_ans'));
% acc = mean(answer == result)


%model_filename = '../model/test.rnn';
%rnn_save_model(model_filename, model);
