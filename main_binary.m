addpath('util');

input_dir = '../feature_1_100/Vec/train';
%input_dir = '../data/test2-c2/train';
train_filename = fullfile(input_dir, 'train_001');
dim = 100;

tic
save_binary_data(train_filename, dim);
toc
