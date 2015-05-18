function save_kaggle_csv(filename, result)

    fprintf('Save %s\n', filename);
    f = fopen(filename, 'w');
    fprintf(f, 'Id,Answer\n');
    
    n = length(result);
    answer = repmat('a', n, 1);
    answer( result == 0 ) = 'a';
    answer( result == 1 ) = 'b';
    answer( result == 2 ) = 'c';
    answer( result == 3 ) = 'd';
    answer( result == 4 ) = 'e';
    
    for i = 1:n
        fprintf(f, '%d,%s\n', i, answer(i));
    end
    
    fclose(f);

end