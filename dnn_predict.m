function y_pred = dnn_predict(model, X)
    
    if( strcmpi(model.opts.activation, 'sigmoid' ) )
        activation      = @sigmoid;
    elseif( strcmpi(model.opts.activation, 'ReLU') )
        activation      = @ReLU;
    else
        error('Unknown activation function %s\n', model.opts.activation);
    end
    
    n_layer = length(model.W);

    a = X;
    for i = 1:n_layer-1
        z = bsxfun(@plus, model.W{i} * a, model.B{i});
        a = activation(z);
    end

    % last layer
    z = bsxfun(@plus, model.W{n_layer} * a, model.B{n_layer});
    y_pred = softmax(z);
    
end