function [y, X, mu, sigma] = rnn_load_data(filename, is_normalize, mu, sigma, mute)
    
    if( ~exist('mute', 'var') )
        mute = 0;
    end
    
    if( ~mute )
        fprintf('Load %s\n', filename);
    end

    D = dlmread(filename);
    yi = D(:, 1);
    Xi = D(:, 2:end);
    clear D;

    
    if( is_normalize )
        if( nargin < 4 )
            % training data normalization
            [Xi, mu, sigma] = normalize_data(Xi);
        else
            % testing data normalization (use mu, sigma from training data)
            [Xi, mu, sigma] = normalize_data(Xi, mu, sigma);
        end
    else
        mu = 0;
        sigma = 1;
    end
    
    split_index = find(yi == 0);
    N = length(split_index);
    y = cell(N, 1);
    X = cell(N, 1);
    
    st = 1;
    for i = 1:N
        ed = split_index(i) - 1;
        y{i} = yi(st+1:ed);
        X{i} = Xi(st:ed-1, 1:end);
        st = split_index(i) + 1;
    end
    
end
% 
% function data_list = list_dir(dir_name)
% 
%     list = dir(dir_name);
%     data_list = {};
%     for i = 1:length(list)
%         if( list(i).name(1) ~= '.' )
%             data_list{end+1} = list(i).name;
%         end
%     end
%         
% end