function benchmark_matrix_multiply(matrix_path, rhs_path, output_fid)
    format long;
    [A, ~,  ~, ~] = mmread(matrix_path);
    n_runs = 10;
    b = load(rhs_path);
    
    [~, name, ~] = fileparts(matrix_path);
    fprintf("Matrix: %s\n", name);
    
    tic;
    for k = 1:n_runs
        x = A * b;
    end
    t = toc;
    % convert to milliseconds %
    t = t * 1000;
    
    fprintf(output_fid, "%s;%d;%f;%f\n", name, n_runs, t, t/n_runs);
end

matsets = [ "cylshell" ];

for matset_idx = 1:length(matsets)
    matset = matsets(matset_idx);
    folder_path = sprintf('../data/%s/', matset);
    files = dir(fullfile(folder_path, '*.mtx'));
    out_fid = fopen(fullfile(sprintf('../output/%s/', matset), 'matlab.csv'), 'w');
    fprintf(out_fid, "matrix_name;num_iter;full_time;avg_time\n");
    for file_idx = 1:length(files)
        file_name = files(file_idx).name;
        [~, matrix_name, ~] = fileparts(file_name);
        rhs_name = sprintf('%s.rhs', matrix_name);
        matrix_path = fullfile(folder_path, file_name);
        rhs_path = fullfile(folder_path, rhs_name);
        benchmark_matrix_multiply(matrix_path, rhs_path, out_fid)
    end
    fclose(out_fid);
end