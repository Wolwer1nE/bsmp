clear, clc; format long 

data = load('../data/4m.txt');

% Matlab uses 1-based indexing
i = data(:,1) + 1;
j = data(:,2) + 1;
values = data(:,3);
n = max([max(i), max(j)]);

%  sparse!
A = sparse(i, j, values, n, n);
spy(A);

% Bandwidth -- rmnove if not needed
[i_idx, j_idx] = find(A);
distances = abs(i_idx - j_idx);
bandwidth = max(distances);
fprintf('Bandwidth: %d\n', bandwidth);

% Saves the image, remove if not needed
filename = 'spy_plot.png';         
saveas(gcf, filename, 'png'); 
