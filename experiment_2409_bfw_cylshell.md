```
Matrix:        data/bfw782b/bfw62a.mtx
RHS:           data/bfw782b/bfw62a.rhs
Tolerance:     1e-6
Max iters:     1000
GMRES restart: 30

Solver     Preconditioner     Status     Converged    Iterations   RelResidual          Time(ms)         Exit    
bicgstab   amg                OK         YES          27           6.11365e-08          61.876           0       
bicgstab   scalar-jacobi      OK         YES          38           7.41601e-07          24.926           0       
bicgstab   block-jacobi       OK         YES          40           5.6246e-07           26.899           0       
bicgstab   none               OK         YES          58           3.68616e-07          29.572           0       
gmres      amg                FAIL       NO           1000         3.31528e-06          434.034          1       
gmres      scalar-jacobi      FAIL       NO           1000         4.98767e-06          301.69           1       
gmres      block-jacobi       FAIL       NO           1000         4.97033e-06          304.666          1       
gmres      none               FAIL       NO           1000         4.75233e-06          288.929          1     
```

```
Matrix:        data/bfw782b/bfw782a.mtx
RHS:           data/bfw782b/bfw782a.rhs
Tolerance:     1e-6
Max iters:     1000
GMRES restart: 30

Solver     Preconditioner     Status     Converged    Iterations   RelResidual          Time(ms)         Exit    
bicgstab   amg                OK         YES          196          7.87924e-07          176.434          0       
bicgstab   scalar-jacobi      FAIL       NO           74           0.313658             30.677           1       
bicgstab   block-jacobi       FAIL       NO           1000         0.000249367          177.176          1       
bicgstab   none               FAIL       NO           882          120161               152.892          1       
gmres      amg                FAIL       NO           1000         2.15778e-05          607.18           1       
gmres      scalar-jacobi      FAIL       NO           1000         2.1836e-05           292.578          1       
gmres      block-jacobi       FAIL       NO           1000         2.12055e-05          295.63           1       
gmres      none               FAIL       NO           1000         0.000404849          292.438          1    
```


```
Matrix:        data/cylshell/s3rmt3m1.mtx
RHS:           data/cylshell/s3rmt3m1.rhs
Tolerance:     1e-6
Max iters:     1000
GMRES restart: 30

Solver     Preconditioner     Status     Converged    Iterations   RelResidual          Time(ms)         Exit    
bicgstab   amg                FAIL       NO           1000         2.96505e+14          1823.08          1       
bicgstab   scalar-jacobi      FAIL       NO           1000         1994.57              191.136          1       
bicgstab   block-jacobi       FAIL       NO           1000         3.16037e+14          167.373          1       
bicgstab   none               FAIL       NO           881          4.71338e+14          157.69           1       
gmres      amg                FAIL       NO           1000         304.713              1100.84          1       
gmres      scalar-jacobi      FAIL       NO           1000         4.15705              312.549          1       
gmres      block-jacobi       FAIL       NO           1000         4.7102               311.649          1       
gmres      none               FAIL       NO           1000         0.753663             308.888          1  
```

```
Matrix:        data/cylshell/s3rmt3m1.mtx
RHS:           data/cylshell/s3rmt3m1.rhs
Tolerance:     1e-6
Max iters:     10000
GMRES restart: 30

Solver     Preconditioner     Status     Converged    Iterations   RelResidual          Time(ms)         Exit    
bicgstab   amg                FAIL       NO           10000        nan                  17498.2          1       
bicgstab   scalar-jacobi      FAIL       NO           853          2.36416e+12          174.726          1       
bicgstab   block-jacobi       FAIL       NO           10000        nan                  1482.48          1       
bicgstab   none               FAIL       NO           1325         6.87681e+14          216.379          1       
gmres      amg                FAIL       NO           10000        538.846              10404.5          1       
gmres      scalar-jacobi      FAIL       NO           10000        3.18018              2836.43          1       
gmres      block-jacobi       FAIL       NO           10000        3.91953              3106.98          1       
gmres      none               FAIL       NO           10000        0.734878             2885.87          1  
```