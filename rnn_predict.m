function [costs, Y_pred] = rnn_predict(model, X, Y)
    
    if( strcmpi(model.opts.activation, 'sigmoid' ) )
        activation      = @sigmoid;
    elseif( strcmpi(model.opts.activation, 'ReLU') )
        activation      = @ReLU;
    else
        error('Unknown activation function %s\n', model.opts.activation);
    end
    
    n_seq = length(X);
    Y_pred = cell(n_seq, 1);
    costs = zeros(n_seq, 1);
        
    for i = 1:n_seq
        
        x = X{i};
        n_data = size(X{i}, 1);
        y_pred = zeros(n_data, model.opts.structure(end));
        
        model.M = zeros(size(model.M)); % clear memory layer

        for j = 1:n_data
            a = x(j, :)';

            z = (model.Wi * a + model.Bi) + ...
                (model.Wm * model.M +  model.Bm);

            a = activation(z);
            
            model.M = a; % store in memory layer

            % last layer
            z = model.Wo * a + model.Bo;
            y = softmax(z);
            y_pred(j, :) = y';
        end
        
        Y_tr = expand_label(Y{i}, model.opts.num_label);
        costs(i) = cross_entropy_loss(Y_tr, y_pred);
        [~, Y_pred{i}] = max(y_pred, [], 2);
    end
    
end