function model = dnn_train(model, X_train, y_train, X_valid, y_valid)
    
    fprintf('DNN training...\n');

    % forward
    N           = size(X_train, 1);
    batch_size  = model.opts.batch_size;
    n_layer     = length(model.W);
    n_batch     = ceil(N/batch_size);
    eta         = model.opts.learning_rate;
    epoch       = model.opts.epoch;
    
    if( strcmpi(model.opts.activation, 'sigmoid' ) )
        activation      = @sigmoid;
        grad_activation = @grad_sigmoid;
    elseif( strcmpi(model.opts.activation, 'ReLU') )
        activation      = @ReLU;
        grad_activation = @grad_ReLU;
    else
        error('Unknown activation function %s\n', model.opts.activation);
    end
    
    fprintf('NN = %s\n', num2str(model.opts.structure));
    fprintf('Batch Size = %d\n', batch_size);
    fprintf('Learning Rate = %f\n', eta);
    fprintf('Activation = %s\n', model.opts.activation);
    fprintf('Total epoch = %d\n', epoch);
    
    
    Y_train = expand_label(y_train, model.opts.num_class);

    for iter = 1:epoch
        index_list = randperm(N); % shuffle
    
        for b = 1:n_batch
            st = (b-1) * batch_size + 1;
            ed = min(b * batch_size, N);
            index = index_list(st:ed);

            x_batch = X_train(index, :)';
            y_batch = Y_train(index, :)';
            
            % forward
            a = x_batch;
            for i = 1:n_layer-1
                model.z{i} = bsxfun(@plus, model.W{i} * a, model.B{i});
                a = activation(model.z{i});
                model.a{i} = a;
            end

            % last layer
            model.z{n_layer} = bsxfun(@plus, model.W{n_layer} * a, model.B{n_layer});
            y_pred = softmax(model.z{n_layer});
            %cost = l2_loss(y_batch, y)
            %cost = cross_entropy_loss(y_batch, y_pred);

            % back propagation
            model.delta{n_layer} = grad_entropy_softmax(y_batch, y_pred);

            for i = n_layer-1:-1:1
                dadz = grad_activation(model.z{i});
                model.delta{i} = ( model.W{i+1}' * model.delta{i+1} ) .* dadz;
            end

            % mini-batch gradient descent
            dCdW = model.delta{1} * x_batch';
            dCdB = sum( model.delta{1}, 2);
            model.W{1} = model.W{1} - eta / batch_size * dCdW;
            model.B{1} = model.B{1} - eta / batch_size * dCdB;
            for i = 2:n_layer
                dCdW = model.delta{i} * model.a{i-1}';
                dCdB = sum( model.delta{i}, 2);
                model.W{i} = model.W{i} - eta / batch_size * dCdW;
                model.B{i} = model.B{i} - eta / batch_size * dCdB;
            end

        end % end of batch
        
        % calculate E_in
        Y_pred = dnn_predict(model, X_train');
        cost = cross_entropy_loss(Y_train', Y_pred);
        [~, y_pred] = max(Y_pred);
        E_in = 1 - mean( y_train == y_pred' );
        
        % calculate E_val
        Y_pred = dnn_predict(model, X_valid');
        [~, y_pred] = max(Y_pred);
        E_val = 1 - mean( y_valid == y_pred' );
        
        fprintf('DNN training: epoch %d, cost = %f,  E_in = %f, E_val = %f\n', ...
                iter, cost, E_in, E_val);

    end % end of epoch

end