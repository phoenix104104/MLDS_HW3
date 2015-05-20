function model = rnn_train(model, X_train, Y_train)
    
    fprintf('RNN training...\n');

    num_dim     = model.opts.num_dim;
    num_class   = model.opts.num_class;
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
    
    if( strcmpi(model.opts.update_grad, 'sgd' ) )
        update_gradient = @sgd;
    elseif( strcmpi(model.opts.update_grad, 'adagrad') )
        update_gradient = @adagrad;
    elseif( strcmpi(model.opts.update_grad, 'rmsprop') )
        update_gradient = @RMSProp;
    else
        error('Unknown gradient update method %s\n', model.opts.update_grad);
    end
    
    fprintf('==============================================\n');
    fprintf('%-20s = %s\n', 'Training data size', num2str(model.opts.num_data));
    fprintf('%-20s = %s\n', 'Data normalization', num2str(model.opts.normalize));
    fprintf('%-20s = %s\n', 'Structure', num2str(model.opts.structure));
    fprintf('%-20s = %s\n', 'Learning Rate', num2str(model.opts.learning_rate));
    fprintf('%-20s = %s\n', 'Momentum', num2str(model.opts.momentum));
    fprintf('%-20s = %s\n', 'Weight decay', num2str(model.opts.weight_decay));
    fprintf('%-20s = %s\n', 'BPTT depth', num2str(model.opts.bptt_depth));
    fprintf('%-20s = %s\n', 'Gradient threshold', num2str(model.opts.gradient_thr));
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
    zo = cell(max_depth, 1);
    delta = cell(max_depth+1, 1);
    
    
    for iter = 1:epoch
        
        index_list = randperm(n_seq); % shuffle
        epoch_time = tic;
        
        for i = 1:n_seq
            
            X = (X_train{index_list(i)});
            Y = (Y_train{index_list(i)});
            Y = expand_label(Y, model.opts.num_class);
            
            M_init = zeros(size(model.M));
            
            n_data = size(X, 1); % number of data in one sequence
            
            for j = max_depth:n_data
                
                % fetch data
                for k = 1:max_depth
                    ptr = j - max_depth + k; % pointer to access X, Y
                    x{k} = X(ptr, :)';
                    y{k} = Y(ptr, :)';
                end
                
                model.M = M_init;
                
                
                % forward
                for curr_depth = 1:max_depth
                    z{curr_depth} = (model.Wi * x{curr_depth} + model.Bi) + ...
                                    (model.Wm * model.M + model.Bm);
                                
                    a{curr_depth} = activation(z{curr_depth});
                    
                    zo{curr_depth} = model.Wo * a{curr_depth} + model.Bo;
                    
                    model.M = a{curr_depth};
                    
                    if( curr_depth == 1 )
                        M_next = model.M;
                    end
                end
                
                % back propagation
                for curr_depth = 1:max_depth
                    y_pred = softmax(zo{curr_depth});

                    %y_pred = zo{curr_depth};
                    
                    delta{curr_depth+1} = grad_entropy_softmax(y{curr_depth}, y_pred);
                    %delta{curr_depth+1} = grad_l2_loss(y{curr_depth}', y_pred);

                    dadz = grad_activation(z{curr_depth});
                    delta{curr_depth} = ( model.Wo' * delta{curr_depth+1} ) .* dadz;

                    for d = curr_depth-1:-1:1
                        dadz = grad_activation(z{d});
                        delta{d} = ( model.Wm' * delta{d+1} ) .* dadz;
                    end

                    model = calculate_gradient(model, x, a, delta, M_init, curr_depth);
                end
                
                model = update_gradient(model);
                
                M_init = M_next;
            
            end % end of data in sequence
            
        end % end of sequence
        
        epoch_time = toc(epoch_time);
        
        % calculate E_in
        Y_pred = rnn_predict(model, X_train);
        cost = 0;
        for i = 1:n_seq
            Y_tr = expand_label(Y_train{i}, model.opts.num_class);
            cost = cost + cross_entropy_loss(Y_tr, Y_pred{i});
            %cost = cost + l2_loss(Y_train{i}, Y_pred{i});
        end
        cost = cost / n_seq;
        model.cost(iter) = gather(cost);
        
        fprintf('RNN training: epoch %d (%.1f s), cost = %f\n', ...
                iter, epoch_time, cost);
        
        if( ~mod(iter, model.opts.epoch_to_save) )
            model_filename = fullfile(model.opts.model_dir, sprintf('epoch%d.rnn', iter));
            rnn_save_model(model_filename, model);
        end
            
    end % end of epoch

end



function model = calculate_gradient(model, x, a, delta, M_init, depth)
    
    thr = model.opts.gradient_thr;
    
    % Wo, Bo
    dCdW = delta{depth+1} * a{depth}';
	dCdB = delta{depth+1};
        
    dCdW = clip_gradient(dCdW, thr);
    dCdB = clip_gradient(dCdB, thr);
    
    model.dWo = model.dWo + dCdW;
    model.dBo = model.dBo + dCdB;
        
    % Wm, Bm
    for d = 2:depth
        dCdW = delta{d} * a{d-1}';
        dCdB = delta{d};
        
        dCdW = clip_gradient(dCdW, thr);
        dCdB = clip_gradient(dCdB, thr);
    
        model.dWm = model.dWm + dCdW;
        model.dBm = model.dBm + dCdB;
    end
    
    dCdW = delta{1} * M_init';
	dCdB = delta{1};
    
    dCdW = clip_gradient(dCdW, thr);
    dCdB = clip_gradient(dCdB, thr);
    
    model.dWm = model.dWm + dCdW;
    model.dBm = model.dBm + dCdB;
        
    % Wi, Bi
    for d = 1:depth
        dCdW = delta{d} * x{d}';
        dCdB = delta{d};
        
        dCdW = clip_gradient(dCdW, thr);
        dCdB = clip_gradient(dCdB, thr);
        
        model.dWi = model.dWi + dCdW;
        model.dBi = model.dBi + dCdB;
    end
    
end

function model = sgd(model)
    
    eta     = model.opts.learning_rate;
    mu      = model.opts.momentum;
    lambda  = 1 - model.opts.weight_decay * eta;
    
    % Wo, Bo
    model.mWo = mu * model.mWo - eta * model.dWo;
    model.mBo = mu * model.mBo - eta * model.dBo;
    
    model.Wo = lambda * model.Wo + model.mWo;
    model.Bo =          model.Bo + model.mBo;
    
    % Wm, Bm
    model.mWm = mu * model.mWm - eta * model.dWm;
    model.mBm = mu * model.mBm - eta * model.dBm;
    
    model.Wm = lambda * model.Wm + model.mWm;
    model.Bm =          model.Bm + model.mBm;
    
    % Wi, Bi
    model.mWi = mu * model.mWi - eta * model.dWi;
    model.mBi = mu * model.mBi - eta * model.dBi;
    
    model.Wi = lambda * model.Wi + model.mWi;
    model.Bi =          model.Bi + model.mBi;
    
    % clear gradient buffer
    model.dWi = zeros(size(model.Wi));
    model.dBi = zeros(size(model.Bi));
    model.dWm = zeros(size(model.Wm));
    model.dBm = zeros(size(model.Bm));
    model.dWo = zeros(size(model.Wo));
    model.dBo = zeros(size(model.Bo));
    
end

function g = clip_gradient(g, thr)

    if( norm(g(:)) > thr )
        g = g / norm(g(:)) * thr;
    end
    
end
