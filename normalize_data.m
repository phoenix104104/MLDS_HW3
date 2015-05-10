function [X, mu, sigma] = normalize_data(X, mu, sigma)
    
    % X in R^(data_size, feature_dimension)
    
    eps = 1e-4;
    if( nargin < 3 )
        mu = mean(X, 1);
        %sigma = std(X, 1);
        %sigma(abs(mu) <= eps) = 1;
        sigma = ones(size(mu));
    end

    X = bsxfun(@minus, X, mu);
    X = bsxfun(@rdivide, X, sigma);
        
end