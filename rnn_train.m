function model = rnn_train(model, X_train, Y_train)
    
    fprintf('RNN training...\n');

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
    fprintf('%-20s = %s\n', 'Structure', num2str(model.opts.structure));
    fprintf('%-20s = %s\n', 'Learning Rate', num2str(model.opts.learning_rate));
    fprintf('%-20s = %s\n', 'Momentum', num2str(model.opts.momentum));
    fprintf('%-20s = %s\n', 'Weight decay', num2str(model.opts.weight_decay));
    fprintf('%-20s = %s\n', 'BPTT depth', num2str(model.opts.bptt_depth));
    fprintf('%-20s = %s\n', 'Gradient threshold', num2str(model.opts.gradient_thr));
    fprintf('%-20s = %s\n', 'Dropout', num2str(dropout_prob));
    fprintf('%-20s = %s\n', 'Activation', model.opts.activation);
    fprintf('%-20s = %s\n', 'Update method', model.opts.update_grad);
    fprintf('%-20s = %d\n', 'Total epoch', epoch);
    fprintf('==============================================\n');
    
    n_seq = length(X_train); % number of sequences
    max_depth = model.opts.bptt_depth;
    
    x  = cell(max_depth, 1);
    y  = cell(max_depth, 1);
    z  = cell(max_depth, 1);
    a  = cell(max_depth, 1);
    m  = cell(max_depth, 1);
    zo = cell(max_depth, 1);
    delta = cell(max_depth+1, 1);
    
    for iter = 1:epoch
        
        index_list = randperm(n_seq); % shuffle
        epoch_time = tic;
        
        for i = 1:n_seq
            X = X_train{index_list(i)};
            Y = Y_train{index_list(i)};
            
            %model = rnn_clear_memory(model);
            M_init = zeros(size(model.M));
            
            n_data = size(X, 1); % number of data in one sequence
            
            
            for j = max_depth:n_data
                
                % fetch data
                for k = 1:max_depth
                    ptr = j - max_depth + k; % pointer to access X, Y
                    x{k} = X(ptr, :);
                    y{k} = Y(ptr, :);
                end
                
                model.M = M_init;
                
                % forward
                for curr_depth = 1:max_depth
                    z{curr_depth} = (model.Wi * x{curr_depth}' + model.Bi) + ...
                                    (model.Wm * model.M + model.Bm);

                    a{curr_depth}  = activation(z{curr_depth}) * (1 - model.opts.dropout_prob);
                    zo{curr_depth} = model.Wo * a{curr_depth} + model.Bo;
                    
                    model.M = a{curr_depth};
                    
                    if( curr_depth == 1 )
                        M_next = model.M;
                    end
                end
                
                % back propagation
                for curr_depth = 1:max_depth
                    y_pred = softmax(zo{curr_depth});
                    
                    delta{curr_depth+1} = grad_cross_entropy_loss(y{curr_depth}', y_pred);

                    dadz = grad_activation(z{curr_depth});
                    delta{curr_depth} = ( model.Wo' * delta{curr_depth+1} ) .* dadz;

                    for d = curr_depth-1:-1:1
                        dadz = grad_activation(z{d});
                        delta{d} = ( model.Wm' * delta{d+1} ) .* dadz;
                    end

                    model = update_grad(model, x, a, delta, M_init, curr_depth);
                end
                
                M_init = M_next;
            
            end % end of data in sequence
            
        end % end of sequence
        
        epoch_time = toc(epoch_time);
        
        % calculate E_in
        Y_pred = rnn_predict(model, X_train);
        cost = 0;
        for i = 1:n_seq
            cost = cost + cross_entropy_loss(Y_train{i}, Y_pred{i});
        end
        
        
        fprintf('DNN training: epoch %d (%.1f s), cost = %f\n', ...
                iter, epoch_time, cost);

    end % end of epoch

end

function model = sgd(model, x, a, delta, M_init, depth)
    
    % mini-batch gradient descent
    % X in R^(1 x feature_dim)
    
    eta     = model.opts.learning_rate;
    lambda  = 1 - model.opts.weight_decay * eta;
    thr     = model.opts.gradient_thr;
    
    % update Wo, Bo
    dCdW = delta{depth+1} * a{depth}';
	dCdB = delta{depth+1};
        
    dCdW = clip_gradient(dCdW, thr);
    dCdB = clip_gradient(dCdB, thr);
    
    model.Wo = lambda * model.Wo - eta * dCdW;
    model.Bo = model.Bo - eta * dCdB;
        
    % update Wm, Bm
    for d = 2:depth
        dCdW = delta{d} * model.a{d-1}';
        dCdB = delta{d};
        
        dCdW = clip_gradient(dCdW, thr);
        dCdB = clip_gradient(dCdB, thr);
    
        model.Wm = lambda * model.Wm - eta * dCdW;
        model.Bm = model.Bm - eta * dCdB;
    end
    
    dCdW = model.delta{1} * M_init';
	dCdB = model.delta{1};
    
    dCdW = clip_gradient(dCdW, thr);
    dCdB = clip_gradient(dCdB, thr);
    
    model.Wm = lambda * model.Wm - eta * dCdW;
    model.Bm = model.Bm - eta * dCdB;
        
    % update Wi, Bi
    for d = 1:depth
        dCdW = model.delta{d} * x{d};
        dCdB = model.delta{d};
        
        dCdW = clip_gradient(dCdW, thr);
        dCdB = clip_gradient(dCdB, thr);
        
        model.Wi = lambda * model.Wi - eta * dCdW;
        model.Bi = model.Bi - eta * dCdB;
    end
    
end

function g = clip_gradient(g, thr)

    if( norm(g(:)) > thr )
        g = g / norm(g(:)) * thr;
    end
    
end
% 
% function model = adagrad(model, X)
%     
%     % mini-batch gradient descent
%     % X in R^(data_size x feature_dim)
%     
%     n_layer     = length(model.W);
%     batch_size  = size(X, 1);
%     eta         = model.opts.learning_rate;
%     lambda      = 1 - model.opts.weight_decay * eta;
%     eps         = 1e-6; % TODO: need tuning
%     
%     dCdW = model.delta{1} * X';
%     dCdB = sum( model.delta{1}, 2);
%     
%     model.sW{1} = model.sW{1} + dCdW.^2;
%     model.sB{1} = model.sB{1} + dCdB.^2;
%     
%     dCdW = dCdW ./ sqrt( model.sW{1} + eps );
%     dCdB = dCdB ./ sqrt( model.sB{1} + eps );
%     
%     model.W{1} = lambda * model.W{1} - eta ./ batch_size .* dCdW;
%     model.B{1} = model.B{1} - eta ./ batch_size .* dCdB;
%     
%     for i = 2:n_layer
%         
%         dCdW = model.delta{i} * model.a{i-1}';
%         dCdB = sum( model.delta{i}, 2);
%         
%         model.sW{i} = model.sW{i} + dCdW.^2;
%         model.sB{i} = model.sB{i} + dCdB.^2;
%         
%         dCdW = dCdW ./ sqrt( model.sW{i} + eps );
%         dCdB = dCdB ./ sqrt( model.sB{i} + eps );
% 
%         model.W{i} = lambda * model.W{i} - eta ./ batch_size .* dCdW;
%         model.B{i} = model.B{i} - eta ./ batch_size .* dCdB;
%     end
% end
% 
% 
% function model = RMSProp(model, X)
%     
%     % mini-batch gradient descent
%     % X in R^(data_size x feature_dim)
%     
%     n_layer     = length(model.W);
%     batch_size  = size(X, 1);
%     eta         = model.opts.learning_rate;
%     alpha       = model.opts.rmsprop_alpha;
%     lambda      = 1 - model.opts.weight_decay * eta;
%     eps         = 1e-6; % TODO: need tuning
%     
%     dCdW = model.delta{1} * X';
%     dCdB = sum( model.delta{1}, 2);
%     
%     model.sW{1} = sqrt( alpha * model.sW{1}.^2 + (1 - alpha) * dCdW.^2 );
%     model.sB{1} = sqrt( alpha * model.sB{1}.^2 + (1 - alpha) * dCdB.^2 );
%     
%     dCdW = dCdW ./ sqrt( model.sW{1} + eps );
%     dCdB = dCdB ./ sqrt( model.sB{1} + eps );
%     
%     model.W{1} = lambda * model.W{1} - eta ./ batch_size .* dCdW;
%     model.B{1} = model.B{1} - eta ./ batch_size .* dCdB;
%     
%     for i = 2:n_layer
%         
%         dCdW = model.delta{i} * model.a{i-1}';
%         dCdB = sum( model.delta{i}, 2);
%     
%         model.sW{i} = sqrt( alpha * model.sW{i}.^2 + (1 - alpha) * dCdW.^2 );
%         model.sB{i} = sqrt( alpha * model.sB{i}.^2 + (1 - alpha) * dCdB.^2 );
%     
%         dCdW = dCdW ./ sqrt( model.sW{i} + eps );
%         dCdB = dCdB ./ sqrt( model.sB{i} + eps );
% 
%         model.W{i} = lambda * model.W{i} - eta ./ batch_size .* dCdW;
%         model.B{i} = model.B{i} - eta ./ batch_size .* dCdB;
% 
%     end
% end