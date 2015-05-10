function g = grad_sigmoid(x)
    
    y = sigmoid(x);
    g = y .* (1 - y);

end