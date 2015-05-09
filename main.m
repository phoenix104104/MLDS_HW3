clear all;

data_dir = '../data';
feature_name = 'mnist-train100.dense.raw';
num_class = 10;

filename = fullfile(data_dir, feature_name);
[Y, X] = dnn_load_data(filename, num_class);

num_data    = size(Y, 1);
num_dim     = size(X, 2);

fprintf('Input data size = %d\n', num_data);
fprintf('Feature dimension = %d\n', num_dim);

opts.learning_rate  = 0.01;
opts.batch_size     = 5;
opts.epoch          = 10;
opts.weight_decay   = 0.0005;
opts.momentum       = 0.9;
opts.hidden         = [128, 100];
opts.structure      = [num_dim, opts.hidden, num_class];
opts.activation     = 'sigmoid';

net = dnn_init(opts);


% forward
a = X(1:opts.batch_size, :)';
n_layer = length(net.opts.hidden);

for i = 1:n_layer
    z = net.W{i} * a + repmat(net.B{i}, [1, opts.batch_size]);
    a = sigmoid(z);
    net.A{i} = a;
end
% last layer
z = net.W{n_layer+1} * a + repmat(net.B{n_layer+1}, [1, opts.batch_size]);
y = softmax(z);
cost = l2_loss(Y(1:opts.batch_size, :)', y)

% backward




% for i = n_layer:-1:1
%     
%     
% 
% 
% end
