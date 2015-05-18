function [result, Y_pred, cost] = rnn_test(model, X_test, Y_test)

    n_seq = length(Y_test);
    
    Y_pred = rnn_predict(model, X_test);
    cost = zeros(n_seq, 1);
    for i = 1:n_seq
        %cost(i) = l2_loss(Y_test{i}, Y_pred{i});
        Y_gt = expand_label(Y_test{i}, model.opts.num_class);
        cost(i) = cross_entropy_loss(Y_gt, Y_pred{i});
    end
    
    [~, result] = min(cost);
end