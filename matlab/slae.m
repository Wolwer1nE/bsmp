clear, clc; format long 
data = load('../data/4m.txt');
b = load('../data/rhs_4m.txt');
i = data(:,1) + 1;
j = data(:,2) + 1;
values = data(:,3);
n = max([max(i), max(j)]);

A = sparse(i, j, values, n, n);

tic
x = A\b;
toc