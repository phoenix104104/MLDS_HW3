function g = grad_softmax(y)

    g = y .* (1 - y);

end