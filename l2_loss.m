function cost = l2_loss(y1, y2)
    
    cost = sum( (y1(:) - y2(:)).^2 );
    
end