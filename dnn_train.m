function model = dnn_train(model, X_train, y_train, X_valid, y_valid)
    
    fprintf('DNN training...\n');

    N               = size(X_train, 1);
    batch_size      = model.opts.batch_size;
    n_layer         = length(model.W);
    n_batch         = ceil(N/batch_size);
    dropout_prob    = model.opts.dropout_prob;
    epoch           = model.opts.epoch;
    
    if( strcmpi(model.opts.activation, 'sigmoid' ) )
        activation      = @sigmoid;
        grad_activation = @grad_sigmoid;
    elseif( strcmpi(model.opts.activation, 'ReLU') )
        activation      = @ReLU;
        grad_activation = @grad_ReLU;
    else
        error('Unknown activation function %s\n', model.opts.activation);
    end
    
    if( strcmpi(model.opts.update_grad, 'sgd' ) )
        update_grad = @sgd;
    elseif( strcmpi(model.opts.update_grad, 'adagrad') )
        update_grad = @adagrad;
    elseif( strcmpi(model.opts.update_grad, 'rmsprop') )
        update_grad = @RMSProp;
    else
        error('Unknown gradient update method %s\n', model.opts.update_grad);
    end
    
    fprintf('==============================================\n');
    fprintf('Structure \t\t= %s\n', num2str(model.opts.structure));
    fprintf('Batch Size \t\t= %d\n', batch_size);
    fprintf('Learning Rate \t= %s\n', num2str(model.opts.learning_rate));
    fprintf('Momentum \t\t= %s\n', num2str(model.opts.momentum));
    fprintf('Weight decay \t= %s\n', num2str(model.opts.weight_decay));
    fprintf('Dropout \t\t= %s\n', num2str(dropout_prob));
    fprintf('Activation \t\t= %s\n', model.opts.activation);
    fprintf('Update method \t= %s\n', model.opts.update_grad);
    fprintf('Total epoch \t= %d\n', epoch);
    fprintf('==============================================\n');
    
    Y_train = expand_label(y_train, model.opts.num_class);

    for iter = 1:epoch
        index_list = randperm(N); % shuffle
    
        for b = 1:n_batch
            st = (b-1) * batch_size + 1;
            ed = min(b * batch_size, N);
            index = index_list(st:ed);

            x_batch = X_train(index, :);
            y_batch = Y_train(index, :);
            
            % forward
            a = x_batch';
            for i = 1:n_layer-1
                model.z{i} = bsxfun(@plus, model.W{i} * a, model.B{i});
                model.m{i} = rand(size(model.z{i})) > (1 - dropout_prob);
                a = activation(model.z{i} .* model.m{i});
                model.a{i} = a;
            end

            % last layer
            model.z{n_layer} = bsxfun(@plus, model.W{n_layer} * a, model.B{n_layer});
            y_pred = softmax(model.z{n_layer});

            % back propagation
            model.delta{n_layer} = grad_entropy_softmax(y_batch', y_pred);

            for i = n_layer-1:-1:1
                dadz = grad_activation(model.z{i});
                model.delta{i} = ( model.W{i+1}' * model.delta{i+1} ) .* dadz;
            end

            model = update_grad(model, x_batch);
            
        end % end of batch
        
        % calculate E_in
        Y_pred = dnn_predict(model, X_train);
        cost = cross_entropy_loss(Y_train, Y_pred);
        [~, y_pred] = max(Y_pred, [], 2);
        E_in = 1 - mean( y_train == y_pred );
        
        % calculate E_val
        Y_pred = dnn_predict(model, X_valid);
        [~, y_pred] = max(Y_pred, [], 2);
        E_val = 1 - mean( y_valid == y_pred );
        
        fprintf('DNN training: epoch %d, cost = %f,  E_in = %f, E_val = %f\n', ...
                iter, cost, E_in, E_val);

    end % end of epoch

end

function model = sgd(model, X)
    
    % mini-batch gradient descent
    % X in R^(data_size x feature_dim)
    
    n_layer     = length(model.W);
    batch_size  = size(X, 1);
    eta         = model.opts.learning_rate;
    mu          = model.opts.momentum;
    lambda      = 1 - model.opts.weight_decay * eta;
    
    dCdW = model.delta{1} * X;
    dCdB = sum( model.delta{1}, 2);
    
    model.mW{1} = mu * model.mW{1} - eta / batch_size * dCdW;
    model.mB{1} = mu * model.mB{1} - eta / batch_size * dCdB;
    
    model.W{1} = lambda * model.W{1} + model.mW{1};
    model.B{1} = model.B{1} + model.mB{1};
    
    for i = 2:n_layer
        
        dCdW = model.delta{i} * model.a{i-1}';
        dCdB = sum( model.delta{i}, 2);
        
        model.mW{i} = mu * model.mW{i} - eta / batch_size * dCdW;
        model.mB{i} = mu * model.mB{i} - eta / batch_size * dCdB;
    
        model.W{i} = lambda * model.W{i} + model.mW{i};
        model.B{i} = model.B{i} + model.mB{i};
    end
end

function model = adagrad(model, X)
    
    % X in R^(data_size x feature_dim)
    
    n_layer     = length(model.W);
    batch_size  = size(X, 1);
    eta         = model.opts.learning_rate;
    lambda      = 1 - model.opts.weight_decay * eta;
    eps         = 1e-6; % TODO: need tuning
    
    dCdW = model.delta{1} * X;
    dCdB = sum( model.delta{1}, 2);
    
    model.sW{1} = model.sW{1} + dCdW.^2;
    model.sB{1} = model.sB{1} + dCdB.^2;
    
    dCdW = dCdW ./ sqrt( model.sW{1} + eps );
    dCdB = dCdB ./ sqrt( model.sB{1} + eps );
    
    model.W{1} = lambda * model.W{1} - eta ./ batch_size .* dCdW;
    model.B{1} = model.B{1} - eta ./ batch_size .* dCdB;
    
    for i = 2:n_layer
        
        dCdW = model.delta{i} * model.a{i-1}';
        dCdB = sum( model.delta{i}, 2);
        
        model.sW{i} = model.sW{i} + dCdW.^2;
        model.sB{i} = model.sB{i} + dCdB.^2;
        
        dCdW = dCdW ./ sqrt( model.sW{i} + eps );
        dCdB = dCdB ./ sqrt( model.sB{i} + eps );

        model.W{i} = lambda * model.W{i} - eta ./ batch_size .* dCdW;
        model.B{i} = model.B{i} - eta ./ batch_size .* dCdB;
    end
end


function model = RMSProp(model, X)
    
    % X in R^(data_size x feature_dim)
    
    n_layer     = length(model.W);
    batch_size  = size(X, 1);
    eta         = model.opts.learning_rate;
    alpha       = model.opts.rmsprop_alpha;
    lambda      = 1 - model.opts.weight_decay * eta;
    eps         = 1e-6; % TODO: need tuning
    
    dCdW = model.delta{1} * X;
    dCdB = sum( model.delta{1}, 2);
    
    model.sW{1} = sqrt( alpha * model.sW{1}.^2 + (1 - alpha) * dCdW.^2 );
    model.sB{1} = sqrt( alpha * model.sB{1}.^2 + (1 - alpha) * dCdB.^2 );
    
    dCdW = dCdW ./ sqrt( model.sW{1} + eps );
    dCdB = dCdB ./ sqrt( model.sB{1} + eps );
    
    model.W{1} = lambda * model.W{1} - eta ./ batch_size .* dCdW;
    model.B{1} = model.B{1} - eta ./ batch_size .* dCdB;
    
    for i = 2:n_layer
        
        dCdW = model.delta{i} * model.a{i-1}';
        dCdB = sum( model.delta{i}, 2);
    
        model.sW{i} = sqrt( alpha * model.sW{i}.^2 + (1 - alpha) * dCdW.^2 );
        model.sB{i} = sqrt( alpha * model.sB{i}.^2 + (1 - alpha) * dCdB.^2 );
    
        dCdW = dCdW ./ sqrt( model.sW{i} + eps );
        dCdB = dCdB ./ sqrt( model.sB{i} + eps );

        model.W{i} = lambda * model.W{i} - eta ./ batch_size .* dCdW;
        model.B{i} = model.B{i} - eta ./ batch_size .* dCdB;

    end
end