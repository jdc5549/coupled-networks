
A = sparse([-2 1;-1 1]);
b_L = [-Inf; -Inf];
b_U = [  2;   3.2];
c   = [ -1;  -2];
x_L = [  0;   0];
x_U = [  3; Inf];
isinteger = [0 1];

disp('Solving without integer variables');
[x,z,status] = osi(c,x_L,x_U,A,b_L,b_U)

disp('Solving with integer variables');
[x,z,status] = osi(c,x_L,x_U,A,b_L,b_U,isinteger)
