clear, clc; format long 

% Load matrix and RHS
data = load('../data/bsmp1.txt');
b = load('../data/bsmp1_rhs.txt');

i = data(:,1) + 1;
j = data(:,2) + 1;
values = data(:,3);
n = max([max(i), max(j)]);

A = sparse(i, j, values, n, n);

fprintf('Solving %dx%d system in MATLAB...\n', n, n);
tic
x_matlab = A\b;
toc

% Load external solution
external_solution_file = '../data/bsmp1_x.txt';
if exist(external_solution_file, 'file')
    x_cpp = load(external_solution_file);
    
    % Check dimensions match
    if length(x_cpp) == length(x_matlab)
        % Compute difference
        diff = x_matlab - x_cpp;
        rel_error = norm(diff) / norm(x_matlab);
        max_abs_error = max(abs(diff));
        
        fprintf('\n=== Comparison with external solution ===\n');
        fprintf('Relative error: %.6e\n', rel_error);
        fprintf('Max absolute error: %.6e\n', max_abs_error);
        
        % Plot comparison
        figure('Name', 'MATLAB vs External Solution');
        
        subplot(2,2,1);
        plot(x_matlab, 'b-', 'LineWidth', 1.5);
        hold on;
        plot(x_cpp, 'r--', 'LineWidth', 1);
        legend('MATLAB', 'External');
        title('Solution Comparison');
        xlabel('Index');
        ylabel('Value');
        grid on;
        
        subplot(2,2,2);
        semilogy(abs(diff), 'k-', 'LineWidth', 1);
        title('Absolute Difference');
        xlabel('Index');
        ylabel('|x_{MATLAB} - x_{External}|');
        grid on;
        
        subplot(2,2,3);
        scatter(x_matlab, x_cpp, 10, 'filled');
        hold on;
        plot([min(x_matlab), max(x_matlab)], [min(x_matlab), max(x_matlab)], 'r--');
        title('Correlation Plot');
        xlabel('MATLAB solution');
        ylabel('External');
        axis equal;
        grid on;
        
        subplot(2,2,4);
        histogram(diff, 50);
        title('Error Distribution');
        xlabel('Error (MATLAB - External)');
        ylabel('Count');
        grid on;
        
        % Verify solution: compute residual for both
        residual_matlab = norm(A*x_matlab - b) / norm(b);
        residual_cpp = norm(A*x_cpp - b) / norm(b);
        
        fprintf('\nResiduals:\n');
        fprintf('  MATLAB:       %.6e\n', residual_matlab);
        fprintf('  External: %.6e\n', residual_cpp);
    else
        warning('External solution size mismatch: expected %d, got %d', length(x_matlab), length(x_cpp));
    end
else
    fprintf('\nExternal solution not found at: %s\n', external_solution_file);
    fprintf('Run: ./bin/r solve -m bicgstab --matrix data/%s.txt --rhs data/rhs_%s.txt\n', dataset, dataset);
end