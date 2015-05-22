function cost = cross_entropy_loss(Y_gt, Y_pred)
    
    % Y in R^(num_class x num_data)
    eps = 1e-4;
    cost = -Y_gt .* log(Y_pred+eps);
    cost = mean( sum(cost, 2) );

end