function y = ReLU(x)
    
    y = (x >= 0) .* x + (x < 0) .* 0;
    
end