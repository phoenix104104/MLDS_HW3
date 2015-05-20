function save_kaggle_csv(filename, result)

    fprintf('Save %s\n', filename);
    f = fopen(filename, 'w');
    fprintf(f, 'Id,Answer\n');
    
    n = length(result);
    answer = repmat('a', n, 1);
    answer( result == 1 ) = 'a';
    answer( result == 2 ) = 'b';
    answer( result == 3 ) = 'c';
    answer( result == 4 ) = 'd';
    answer( result == 5 ) = 'e';
    
    for i = 1:n
        fprintf(f, '%d,%s\n', i, answer(i));
    end
    
    fclose(f);

end