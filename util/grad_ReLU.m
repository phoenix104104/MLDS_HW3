function g = grad_ReLU(x)
    
    g = (x >= 0) * 1 + (x < 0) * 0;
    
end