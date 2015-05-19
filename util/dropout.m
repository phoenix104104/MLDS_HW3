function y = dropout(x, dropout_prob)
    
    M = rand(size(x));
    y = (M > dropout_prob) .* x;
    
end