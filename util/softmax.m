function y = softmax(x)
    
    x = exp(x);
    z = sum(x, 1);
    y = bsxfun(@rdivide, x, z);

end