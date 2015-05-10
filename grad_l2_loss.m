function g = grad_l2_loss(y_gt, y_pred)
    
    g = 2 * (y_gt - y_pred);

end