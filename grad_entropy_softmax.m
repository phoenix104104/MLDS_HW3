function g = grad_entropy_softmax(y_gt, y_pred)

    %g = (y_gt == 1) .* (y_pred - 1) + (y_gt == 0) .* y_pred;
    g = y_pred - y_gt;

end