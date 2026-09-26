

1000 итераций, случайный rhs:
```
Solver     Preconditioner     Status     Converged    Iterations   RelResidual          Time(ms)         Exit    
bicgstab   amg                FAIL       NO           1000         0.407729             10593            1       
bicgstab   scalar-jacobi      FAIL       NO           643          36.0973              367.29           1       
bicgstab   block-jacobi       FAIL       NO           759          0.0148048            445.938          1       
bicgstab   none               FAIL       NO           1000         0.0204603            554.859          1       
gmres      amg                FAIL       NO           1000         0.906189             6246.37          1       
gmres      scalar-jacobi      FAIL       NO           1000         0.0117868            516.266          1       
gmres      block-jacobi       FAIL       NO           1000         0.0114845            521.254          1       
gmres      none               FAIL       NO           1000         0.0160442            517.348          1  
```


10000 итераций, случайный rhs
```
Solver     Preconditioner     Status     Converged    Iterations   RelResidual          Time(ms)         Exit    
bicgstab   amg                FAIL       NO           3421         1.12585e+15          33453.7          1       
bicgstab   scalar-jacobi      FAIL       NO           912          2.1191               504.218          1       
bicgstab   block-jacobi       FAIL       NO           1308         1.32448e+15          710.444          1       
bicgstab   none               FAIL       NO           1700         6.17884e+10          918.695          1       
gmres      amg                FAIL       NO           10000        0.914623             51179.2          1       
gmres      scalar-jacobi      FAIL       NO           10000        0.0107583            4934.09          1       
gmres      block-jacobi       FAIL       NO           10000        0.01079              4949.3           1       
gmres      none               FAIL       NO           10000        0.00969569           4924.61          1    
```

1000 итераций, rhs из задачи:
```
bicgstab   amg                FAIL       NO           306          1.56617              4132.64          1       
bicgstab   scalar-jacobi      FAIL       NO           265          143.356              166.349          1       
bicgstab   block-jacobi       FAIL       NO           246          0.505574             158.239          1       
bicgstab   none               FAIL       NO           1000         1.73233              551.23           1       
gmres      amg                FAIL       NO           1000         6.95364              6232.63          1       
gmres      scalar-jacobi      FAIL       NO           1000         0.153388             521.776          1       
gmres      block-jacobi       FAIL       NO           1000         0.151344             525.893          1       
gmres      none               FAIL       NO           1000         0.144837             519.554          1   
```

10000 итераций, rhs из задачи:
```
Solver     Preconditioner     Status     Converged    Iterations   RelResidual          Time(ms)         Exit    
bicgstab   amg                FAIL       NO           210          9.68127              3244.11          1       
bicgstab   scalar-jacobi      FAIL       NO           479          2.3889               276.974          1       
bicgstab   block-jacobi       FAIL       NO           334          574.705              203.726          1       
bicgstab   none               FAIL       NO           1359         5.14828e+12          740.078          1       
gmres      amg                FAIL       NO           10000        6.95723              51604.2          1       
gmres      scalar-jacobi      FAIL       NO           10000        0.143834             5143.84          1       
gmres      block-jacobi       FAIL       NO           10000        0.151076             4965.18          1       
gmres      none               FAIL       NO           10000        0.132568             4940.6           1     
```