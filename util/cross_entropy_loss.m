function cost = cross_entropy_loss(Y_gt, Y_pred)
    
    % Y in R^(num_class x num_data)
    
    cost = -Y_gt .* log(Y_pred);
    cost = mean( sum(cost, 2) );

end