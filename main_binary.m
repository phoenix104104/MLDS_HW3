addpath('util');

%input_dir = '../feature_15_100/Vec/train';
input_dir = '../data/test2-c2/train';
train_filename = fullfile(input_dir, 'train');
dim = 4;
normalize = 0;

tic
save_binary_data(train_filename, dim);
toc