function g = grad_cross_entropy_loss(y_gt, y_pred)

    g = y_gt .* (1 ./ y_pred);

end