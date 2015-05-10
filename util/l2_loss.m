function cost = l2_loss(y_gt, y_pred)
    
    % y \in R^(Dim x BatchSize)

    cost = (y_gt - y_pred).^2;
    cost = mean( sum(cost, 1) );
    
end