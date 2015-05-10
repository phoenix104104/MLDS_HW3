function cost = cross_entropy_loss(y_gt, y_pred)
    
    cost = -y_gt .* log(y_pred);
    cost = mean( sum(cost, 1) );

end