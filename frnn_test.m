function [result, Y_pred, costs] = frnn_test(model, X_test, Y_test)
    
    [costs, Y_pred] = frnn_predict(model, X_test, Y_test);
    [~, result] = min(costs);
    
end