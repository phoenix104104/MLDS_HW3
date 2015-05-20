function [y, X] = rnn_load_data(file_dir, filename_list, mute)
    
    if( ~exist('mute', 'var') )
        mute = 0;
    end
    
    yi = [];
    Xi = [];
    for i = 1:length(filename_list)
        filename = fullfile(file_dir, filename_list{i});
        if( ~mute )
            fprintf('Load %s\n', filename);
        end
    
        D = dlmread(filename);
        yi = [yi; D(:, 1)];
        Xi = [Xi; D(:, 2:end)];
    end
    clear D;

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
