function [Y_train, X_train, Y_valid, X_valid] = ...
            load_hw2_data(train_filename, valid_filename, train_listname, valid_listname)
    
    f = fopen(train_listname);
    C = textscan(f, '%d');
    train_count_list = C{1};
    fclose(f);
    
    f = fopen(valid_listname);
    C = textscan(f, '%d');
    valid_count_list = C{1};
    fclose(f);


    [yt, Xt] = load_dense_data(train_filename);
    [yv, Xv] = load_dense_data(valid_filename);
    
    fprintf('Data normalization...\n');
    [Xt, mu, sigma] = normalize_data(Xt);
    [Xv, mu, sigma] = normalize_data(Xv, mu, sigma);
    
    fprintf('Split data into sequence...\n');
    [Y_train, X_train] = split_data(yt, Xt, train_count_list);
    [Y_valid, X_valid] = split_data(yv, Xv, valid_count_list);

end

function [Y_seq, X_seq] = split_data(y, X, count_list)

    n_seq = length(count_list);
    Y_seq = cell(n_seq, 1);
    X_seq = cell(n_seq, 1);
    st = 1;
    ed = 0;
    for i = 1:n_seq
        if( i > 1 )
            st = st + count_list(i-1);
        end
        ed = ed + count_list(i);
        
        Y_seq{i} = y(st:ed, :);
        X_seq{i} = X(st:ed, :);
        
    end

end
% function [y, X] = load_hw2_data(data_dir, data_list, feature_name)
% 
%     N = length(data_list);
%     y = cell(N, 1);
%     X = cell(N, 1);
%     fprintf('Load data from %s\n', data_dir);
%         
%     for i = 1:N
%         filename = fullfile(data_dir, sprintf('%s.%s', data_list{i}, feature_name));
%         [y{i}, X{i}] = load_dense_data(filename, 1);
%     end
%     
% end