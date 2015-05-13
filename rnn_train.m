function model = rnn_train(model, X_train, y_train, X_valid, y_valid)
    
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
    depth = model.opts.bptt_depth;
    
    xm = cell(depth, 1);
    
    for iter = 1:epoch
        
        index_list = randperm(n_seq); % shuffle
        epoch_time = tic;
        
        for i = 1:n_seq
            X = X_train{index_list(i)};
            y = y_train{index_list(i)};
            Y = expand_label(y, model.opts.num_class);
            
            model = rnn_clear_memory(model);
            
            n_data = size(X, 1); % number of data in one sequence
            
            ptr = [1, 2, 3];
            
            % --- process first data
            x = X(1, :);
            y_gt = Y(1, :);
            
            % forward
            z = (model.Wi * x' + model.Bi) + ...
                (model.Wm * model.M + model.Bm);

            a = activation(z);
            a = a * (1 - model.opts.dropout_prob);

            % save for back-propagation
            model.S{1} = model.M;
            model.z{1} = z;
            model.a{1} = a;
            xm{1} = x;
            
            % store in memory layer
            model.M = a;
                
            % last layer
            z = model.Wo * a + model.Bo;
            y_pred = softmax(z);
                
                
            % back propagation
            model.delta{2} = grad_entropy_softmax(y_gt', y_pred);
            
            dadz = grad_activation(model.z{1});
            model.delta{1} = ( model.Wo' * model.delta{2} ) .* dadz;
            
            model = update_grad(model, xm, 1, ptr);
            % -------------------------------------------------------------
            
            
            % --- process second data
            x = X(2, :);
            y_gt = Y(2, :);
            
            % forward
            z = (model.Wi * x' + model.Bi) + ...
                (model.Wm * model.M + model.Bm);

            a = activation(z);
            a = a * (1 - model.opts.dropout_prob);
            
            % save for back-propagation
            model.S{2} = model.M;
            model.z{2} = z;
            model.a{2} = a;
            xm{2} = x;
            
            % store in memory layer
            model.M = a;
            
            % last layer
            z = model.Wo * a + model.Bo;
            y_pred = softmax(z);
                
                
            % back propagation
            model.delta{3} = grad_entropy_softmax(y_gt', y_pred);
            
            dadz = grad_activation(model.z{2});
            model.delta{2} = ( model.Wo' * model.delta{3} ) .* dadz;
            
            dadz = grad_activation(model.z{1});
            model.delta{1} = ( model.Wm' * model.delta{2} ) .* dadz;
            
            model = update_grad(model, xm, 2, ptr);
            
            % -------------------------------------------------------------
            
            
            
            ptr = [2, 3, 1];
            for j = 3:n_data
                ptr = mod(ptr+1, 3)+1; % when j=3, ptr = [1, 2, 3]
                
                x = X(j, :);
                y_gt = Y(j, :);
                
                % forward
                z = (model.Wi * x' + model.Bi) + ...
                    (model.Wm * model.M + model.Bm);

                a = activation(z);
                a = a * (1 - model.opts.dropout_prob);
                
                % save for back-propagation
                model.S{ptr(depth)} = model.M;
                model.z{ptr(depth)} = z;
                model.a{ptr(depth)} = a;
                xm{ptr(depth)} = x;
                
                % store in memory layer
                model.M = a;
                
                
                % last layer
                z = model.Wo * a + model.Bo;
                y_pred = softmax(z);
                
                
                % back propagation
                model.delta{depth+1} = grad_entropy_softmax(y_gt', y_pred);
                
                dadz = grad_activation(model.z{ptr(depth)});
                model.delta{depth} = ( model.Wo' * model.delta{depth+1} ) .* dadz;
                    
                for d = depth-1:-1:1
                    dadz = grad_activation(model.z{ptr(d)});
                    model.delta{d} = ( model.Wm' * model.delta{d+1} ) .* dadz;
                end
                
                model = update_grad(model, xm, depth, ptr);
                
                       
            end % end of data in sequence
            
        end % end of sequence
        
        epoch_time = toc(epoch_time);
        
        % calculate E_in
        [y_label, Y_pred] = rnn_predict(model, X_train);
        cost = 0;
        acc = 0;
        for i = 1:n_seq
            Y_gt = expand_label(y_train{i}, model.opts.num_class);
            cost = cost + cross_entropy_loss(Y_gt, Y_pred{i});
            acc = acc + mean( y_train{i} == y_label{i} );
        end
        
        E_in = 1 - acc / n_seq;
        
        % calculate E_val
        % calculate E_in
        [y_label, Y_pred] = rnn_predict(model, X_valid);
        acc = 0;
        for i = 1:length(X_valid)
            acc = acc + mean( y_valid{i} == y_label{i} );
        end
        
        E_val = 1 - acc / length(X_valid);
        
        fprintf('DNN training: epoch %d (%.1f s), cost = %f,  E_in = %f, E_val = %f\n', ...
                iter, epoch_time, cost, E_in, E_val);

    end % end of epoch

end

function model = sgd(model, xm, depth, ptr)
    
    % mini-batch gradient descent
    % X in R^(1 x feature_dim)
    
    eta     = model.opts.learning_rate;
    lambda  = 1 - model.opts.weight_decay * eta;
    thr     = model.opts.gradient_thr;
    
    % update Wo, Bo
    dCdW = model.delta{depth+1} * model.a{depth}';
	dCdB = model.delta{depth+1};
        
    dCdW = clip_gradient(dCdW, thr);
    dCdB = clip_gradient(dCdB, thr);
    
    model.Wo = lambda * model.Wo - eta * dCdW;
    model.Bo = model.Bo - eta * dCdB;
        
    % update Wm, Bm
    for d = 2:depth
        dCdW = model.delta{ptr(d)} * model.a{ptr(d-1)}';
        dCdB = model.delta{ptr(d)};
        
        dCdW = clip_gradient(dCdW, thr);
        dCdB = clip_gradient(dCdB, thr);
    
        model.Wm = lambda * model.Wm - eta * dCdW;
        model.Bm = model.Bm - eta * dCdB;
    end
    
    dCdW = model.delta{ptr(1)} * model.S{ptr(1)}';
	dCdB = model.delta{ptr(1)};
    
    dCdW = clip_gradient(dCdW, thr);
    dCdB = clip_gradient(dCdB, thr);
    
    model.Wm = lambda * model.Wm - eta * dCdW;
    model.Bm = model.Bm - eta * dCdB;
        
    % update Wi, Bi
    for d = 1:depth
        dCdW = model.delta{ptr(d)} * xm{ptr(d)};
        dCdB = model.delta{ptr(d)};
        
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