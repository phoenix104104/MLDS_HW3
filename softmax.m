function y = softmax(x)
    
    z = sum(exp(x), 1);
    y = bsxfun(@rdivide, exp(x), z);

end