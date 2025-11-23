clear, clc; format long 
data = load('../data/100000.txt');
n = 1;
i = data(:,1) + 1;
j = data(:,2) + 1;
values = data(:,3);
n = max([max(i), max(j)]);

A = sparse(i, j, values, n, n);
b = load('../data/rhs_100000.txt');
tic
for k = 1:n
    x = A*b;
end
t = toc;
t/n*1
