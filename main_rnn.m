addpath('util');

%list_name = '../hw2_feature/list/train.list';
f = fopen(list_name);
C = textscan(f, '%s');
data_list = C{1};
fclose(f);

data_dir = '../hw2_feature';
feature_name = 'fbank';

%train_list = data_list(1:10);
%valid_list = data_list(end-10:end);
train_filename = fullfile(data_dir, sprintf('train400p.%s', feature_name));
valid_filename = fullfile(data_dir, sprintf('valid400p.%s', feature_name));
train_list_name = '../hw2_feature/list/train400p.count';
valid_list_name = '../hw2_feature/list/valid400p.count';

% tic
% %[y_train, X_train] = load_hw2_data(data_dir, train_list, feature_name);
% %[y_valid, X_valid] = load_hw2_data(data_dir, valid_list, feature_name);
% [y_train, X_train, y_valid, X_valid] ...
%     = load_hw2_data(train_filename, valid_filename, train_list_name, valid_list_name);
% toc

num_class = 48;
num_data = length(y_train);
num_dim = size(X_train{1}, 2);
fprintf('Input data size = %d\n', num_data);
fprintf('Feature dimension = %d\n', num_dim);

opts.num_class      = num_class;
opts.learning_rate  = 0.01;
opts.epoch          = 100;
opts.weight_decay   = 0;
opts.momentum       = 0.9;
opts.rmsprop_alpha  = 0.9;
opts.dropout_prob   = 0.0;
opts.bptt_depth     = 3;
opts.gradient_thr   = 1; %%
opts.hidden         = [256];
opts.structure      = [num_dim, opts.hidden, num_class];
opts.activation     = 'sigmoid';
opts.update_grad    = 'sgd';

sample = randi(length(X_train), 500, 1);

model = rnn_init(opts);
model = rnn_train(model, X_train(sample), y_train(sample), X_valid, y_valid);


