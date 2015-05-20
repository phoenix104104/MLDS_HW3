function [result, Y_pred, costs] = rnn_test(model, X_test, Y_test)
    
    [costs, Y_pred] = rnn_predict(model, X_test, Y_test);
    [~, result] = min(costs);
    
end