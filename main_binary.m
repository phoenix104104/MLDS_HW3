addpath('util');

input_dir = '../feature_1_100_reduce/Vec/train';
%input_dir = '../data/test2-c2/train';

dim = 100;
data_list = 1:10;

for i = 1:length(data_list)
    train_filename = fullfile(input_dir, sprintf('train_%03d', data_list(i)));

    tic
    save_binary_data(train_filename, dim);
    toc

end
