function y = softmax(x)
    
    m = max(x);
    x = bsxfun(@minus, x, m);
    
    x = exp(x);
    z = sum(x, 1);
    y = bsxfun(@rdivide, x, z);

end