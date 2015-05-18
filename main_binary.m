addpath('util');

input_dir = '../feature_15_100/Vec/train';
train_filename = fullfile(input_dir, 'train_001');
dim = 100;
normalize = 0;

tic
save_binary_data(train_filename, dim);
toc