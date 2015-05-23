function model = frnn_train_all(model, train_dir, train_namelist)
    
    fprintf('RNN training...\n');
    
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
    elseif( strcmpi(model.opts.update_grad, 'rmsprop') )
        update_gradient = @RMSProp;
    else
        error('Unknown gradient update method %s\n', model.opts.update_grad);
    end
    
    fprintf('==============================================\n');
    fprintf('%-20s = %s\n', 'Training data size', num2str(model.opts.num_data));
    fprintf('%-20s = %s\n', 'Feature dimension', num2str(model.opts.num_dim));
    fprintf('%-20s = %s\n', 'Class', num2str(model.opts.num_class));
    fprintf('%-20s = %s\n', 'Structure', num2str(model.opts.structure));
    fprintf('%-20s = %s\n', 'Learning Rate', num2str(model.opts.learning_rate));
    fprintf('%-20s = %s\n', 'Momentum', num2str(model.opts.momentum));
    fprintf('%-20s = %s\n', 'Weight decay', num2str(model.opts.weight_decay));
    fprintf('%-20s = %s\n', 'BPTT depth', num2str(model.opts.bptt_depth));
    fprintf('%-20s = %s\n', 'Gradient threshold', num2str(model.opts.gradient_thr));
    fprintf('%-20s = %s\n', 'Activation', model.opts.activation);
    fprintf('%-20s = %s\n', 'Update method', model.opts.update_grad);
    fprintf('%-20s = %d\n', 'Total epoch', model.opts.epoch);
    fprintf('==============================================\n');
    
    max_depth = model.opts.bptt_depth;
    
    x  = cell(max_depth, 1);
    y  = cell(max_depth, 1);
    c  = cell(max_depth, 1);
    z  = cell(max_depth, 1);
    a  = cell(max_depth, 1);
    zo = cell(max_depth, 1);
    zc = cell(max_depth, 1);
    Wo = cell(max_depth, 1);
    Bo = cell(max_depth, 1);
    y_idx = cell(max_depth, 1);
    delta_o = cell(max_depth+1, 1);
    delta_c = cell(max_depth+1, 1);
    
    
    for iter = 1:model.opts.epoch
        
        for batch = 1:length(train_namelist)
            
            [Y_train, X_train] = rnn_load_binary_data(train_dir, train_namelist{batch});
            C_train = map_y_to_class(Y_train, model.class_map);
            
            n_seq = length(X_train); % number of sequences
            index_list = randperm(n_seq); % shuffle
            
        
            fprintf('RNN training: ');
            epoch_time = tic;

            for i = 1:n_seq

                X = (X_train{index_list(i)});
                Y = (Y_train{index_list(i)});
                Y_ex = expand_label(Y, model.opts.num_label);

                C = (C_train{index_list(i)});
                C_ex = expand_label(C, model.opts.num_class);

                M_init = zeros(size(model.M));

                n_data = size(X, 1); % number of data in one sequence

                for j = max_depth:n_data

                    % fetch data
                    for k = 1:max_depth
                        ptr = j - max_depth + k; % pointer to access X, Y

                        y_idx{k} = model.class{C(ptr)}; % index of y with the same class
                        x{k} = X(ptr, :)';
                        c{k} = C_ex(ptr, :)';
                        y{k} = Y_ex(ptr, y_idx{k}(:))';
                        Wo{k} = model.Wo(y_idx{k}(:), :);
                        Bo{k} = model.Bo(y_idx{k}(:));
                    end

                    model.M = M_init;

                    % forward
                    for curr_depth = 1:max_depth
                        z{curr_depth} = (model.Wi * x{curr_depth} + model.Bi) + ...
                                        (model.Wm * model.M + model.Bm);

                        a{curr_depth} = activation(z{curr_depth});

                        zo{curr_depth} = model.Wo * a{curr_depth} + model.Bo;
                        zc{curr_depth} = model.Wc * a{curr_depth} + model.Bc;

                        model.M = a{curr_depth};

                        if( curr_depth == 1 )
                            M_next = model.M;
                        end
                    end

                    % back propagation
                    for curr_depth = 1:max_depth

                        % Wo part
                        y_pred = softmax(zo{curr_depth});

                        y_pred = y_pred(y_idx{curr_depth});

                        delta_o{curr_depth+1} = grad_entropy_softmax(y{curr_depth}, y_pred);

                        dadz = grad_activation(z{curr_depth});
                        delta_o{curr_depth} = ( Wo{curr_depth}' * delta_o{curr_depth+1} ) .* dadz;

                        for d = curr_depth-1:-1:1
                            dadz = grad_activation(z{d});
                            delta_o{d} = ( model.Wm' * delta_o{d+1} ) .* dadz;
                        end

                        % Wc part
                        c_pred = softmax(zc{curr_depth});

                        delta_c{curr_depth+1} = grad_entropy_softmax(c{curr_depth}, c_pred);

                        dadz = grad_activation(z{curr_depth});
                        delta_c{curr_depth} = ( model.Wc' * delta_c{curr_depth+1} ) .* dadz;

                        for d = curr_depth-1:-1:1
                            dadz = grad_activation(z{d});
                            delta_c{d} = ( model.Wm' * delta_c{d+1} ) .* dadz;
                        end

                        model = calculate_gradient(model, x, a, delta_o, delta_c, M_init, curr_depth);
                    end

                    model = update_gradient(model, y_idx);

                    M_init = M_next;

                end % end of data in sequence

            end % end of sequence

            epoch_time = toc(epoch_time);

            % calculate E_in
            costs = frnn_predict(model, X_train, Y_train);

            model.cost(iter) = mean(costs);

            fprintf('epoch %d (%.1f s), cost = %f\n', ...
                    iter, epoch_time, model.cost(iter));

        end % end of batch

        if( ~mod(iter, model.opts.epoch_to_save) )
            model_filename = fullfile(model.opts.model_dir, sprintf('epoch%d.rnn', iter));
            rnn_save_model(model_filename, model);
        end
                    
    end % end of epoch

end



function model = calculate_gradient(model, x, a, delta_o, delta_c, M_init, depth)
    
    thr = model.opts.gradient_thr;
    
    % Wo, Bo
    dCdW = delta_o{depth+1} * a{depth}';
	dCdB = delta_o{depth+1};
        
    dCdW = clip_gradient(dCdW, thr);
    dCdB = clip_gradient(dCdB, thr);
    
    %model.dWo(y_idx{depth}, :) = model.dWo(y_idx{depth}, :) + dCdW;
    %model.dBo(y_idx{depth}, :) = model.dBo(y_idx{depth}, :) + dCdB;
    model.dWo{depth} = dCdW;
    model.dBo{depth} = dCdB;
    
    % Wc, Bc
    dCdW = delta_c{depth+1} * a{depth}';
	dCdB = delta_c{depth+1};
    
    dCdW = clip_gradient(dCdW, thr);
    dCdB = clip_gradient(dCdB, thr);
    
    model.dWc = model.dWc + dCdW;
    model.dBc = model.dBc + dCdB;
        
    % Wm, Bm
    for d = 2:depth
        dCdW = delta_o{d} * a{d-1}';
        dCdB = delta_o{d};
        
        dCdW = clip_gradient(dCdW, thr);
        dCdB = clip_gradient(dCdB, thr);
    
        model.dWm = model.dWm + dCdW;
        model.dBm = model.dBm + dCdB;
        
        dCdW = delta_c{d} * a{d-1}';
        dCdB = delta_c{d};
        
        dCdW = clip_gradient(dCdW, thr);
        dCdB = clip_gradient(dCdB, thr);
    
        model.dWm = model.dWm + dCdW;
        model.dBm = model.dBm + dCdB;
    end
    
    dCdW = delta_o{1} * M_init';
	dCdB = delta_o{1};
    
    dCdW = clip_gradient(dCdW, thr);
    dCdB = clip_gradient(dCdB, thr);
    
    model.dWm = model.dWm + dCdW;
    model.dBm = model.dBm + dCdB;
    
    dCdW = delta_c{1} * M_init';
	dCdB = delta_c{1};
    
    dCdW = clip_gradient(dCdW, thr);
    dCdB = clip_gradient(dCdB, thr);
    
    model.dWm = model.dWm + dCdW;
    model.dBm = model.dBm + dCdB;
        
    % Wi, Bi
    for d = 1:depth
        dCdW = delta_o{d} * x{d}';
        dCdB = delta_o{d};
        
        dCdW = clip_gradient(dCdW, thr);
        dCdB = clip_gradient(dCdB, thr);
        
        model.dWi = model.dWi + dCdW;
        model.dBi = model.dBi + dCdB;
        
        dCdW = delta_c{d} * x{d}';
        dCdB = delta_c{d};
        
        dCdW = clip_gradient(dCdW, thr);
        dCdB = clip_gradient(dCdB, thr);
        
        model.dWi = model.dWi + dCdW;
        model.dBi = model.dBi + dCdB;
    end
    
end

function n = checkNaN(A)
    
    n = sum(isnan(A(:)));
    
end

function model = sgd(model, y_idx)
    
    eta     = model.opts.learning_rate;
    mu      = model.opts.momentum;
    lambda  = 1 - model.opts.weight_decay * eta;
    d       = model.opts.bptt_depth;
    ds      = d * (d + 1) / 2 * 2;
    thr     = model.opts.gradient_thr;
    
    % Wo, Bo
    for i = 1:d
        model.mWo(y_idx{i}, :) = mu * model.mWo(y_idx{i}, :) - eta * model.dWo{i};
        model.mBo(y_idx{i}, :) = mu * model.mBo(y_idx{i}, :) - eta * model.dBo{i};

        model.Wo(y_idx{i}, :) = lambda * model.Wo(y_idx{i}, :) + model.mWo(y_idx{i}, :);
        model.Bo(y_idx{i}, :) = lambda * model.Bo(y_idx{i}, :) + model.mBo(y_idx{i}, :);

    end
%     model.mWo = mu * model.mWo - eta * mWo_next / d;
%     model.mBo = mu * model.mBo - eta * mBo_next / d;
% 
%     model.Wo = lambda * model.Wo + model.mWo;
%     model.Bo =          model.Bo + model.mBo;
        
    % Wc, Bc
    model.mWc = mu * model.mWc - eta * model.dWc / d;
    model.mBc = mu * model.mBc - eta * model.dBc / d;
    
    model.Wc = lambda * model.Wc + model.mWc;
    model.Bc = lambda * model.Bc + model.mBc;

    % Wm, Bm
    model.mWm = mu * model.mWm - eta * model.dWm / ds;
    model.mBm = mu * model.mBm - eta * model.dBm / ds;
    
    model.Wm = lambda * model.Wm + model.mWm;
    model.Bm = lambda * model.Bm + model.mBm;
    
    % Wi, Bi
    model.mWi = mu * model.mWi - eta * model.dWi / ds;
    model.mBi = mu * model.mBi - eta * model.dBi / ds;
    
    model.Wi = lambda * model.Wi + model.mWi;
    model.Bi = lambda * model.Bi + model.mBi;
    
    % clear gradient buffer
    model.dWi = zeros(size(model.Wi));
    model.dBi = zeros(size(model.Bi));
    model.dWm = zeros(size(model.Wm));
    model.dBm = zeros(size(model.Bm));
    %model.dWo = zeros(size(model.Wo));
    %model.dBo = zeros(size(model.Bo));
    model.dWc = zeros(size(model.Wc));
    model.dBc = zeros(size(model.Bc));
    
end

function g = clip_gradient(g, thr)

    if( norm(g(:)) > thr )
        g = g / norm(g(:)) * thr;
    end
    
end


function model = RMSProp(model)
    
    eta     = model.opts.learning_rate;
    mu      = model.opts.momentum;
    alpha   = model.opts.rmsprop_alpha;
    lambda  = 1 - model.opts.weight_decay * eta;
    eps     = 1e-4; % TODO: need tuning
    thr     = model.opts.gradient_thr;
    
    % Wo, Bo
    model.sWo = sqrt( alpha * model.sWo.^2 + (1 - alpha) * model.dWo.^2 );
    model.sBo = sqrt( alpha * model.sBo.^2 + (1 - alpha) * model.dBo.^2 );
    
    model.dWo = model.dWo ./ sqrt( model.sWo + eps );
    model.dBo = model.dBo ./ sqrt( model.sBo + eps );
    
    model.dWo = clip_gradient(model.dWo, thr);
    model.dBo = clip_gradient(model.dBo, thr);
    
    model.mWo = mu * model.mWo - eta * model.dWo;
    model.mBo = mu * model.mBo - eta * model.dBo;
    
    model.Wo = lambda * model.Wo + model.mWo;
    model.Bo =          model.Bo + model.mBo;
    
    % Wm, Bm
    model.sWm = sqrt( alpha * model.sWm.^2 + (1 - alpha) * model.dWm.^2 );
    model.sBm = sqrt( alpha * model.sBm.^2 + (1 - alpha) * model.dBm.^2 );
    
    model.dWm = model.dWm ./ sqrt( model.sWm + eps );
    model.dBm = model.dBm ./ sqrt( model.sBm + eps );
    
    model.dWm = clip_gradient(model.dWm, thr);
    model.dBm = clip_gradient(model.dBm, thr);
    
    model.mWm = mu * model.mWm - eta * model.dWm;
    model.mBm = mu * model.mBm - eta * model.dBm;
    
    model.Wm = lambda * model.Wm + model.mWm;
    model.Bm =          model.Bm + model.mBm;
    
    % Wi, Bi
    model.sWi = sqrt( alpha * model.sWi.^2 + (1 - alpha) * model.dWi.^2 );
    model.sBi = sqrt( alpha * model.sBi.^2 + (1 - alpha) * model.dBi.^2 );
    
    model.dWi = model.dWi ./ sqrt( model.sWi + eps );
    model.dBi = model.dBi ./ sqrt( model.sBi + eps );
    
    model.dWi = clip_gradient(model.dWi, thr);
    model.dBi = clip_gradient(model.dBi, thr);
    
    model.mWi = mu * model.mWi - eta * model.dWi;
    model.mBi = mu * model.mBi - eta * model.dBi;
    
    model.Wi = lambda * model.Wi + model.mWi;
    model.Bi =          model.Bi + model.mBi;
    
end