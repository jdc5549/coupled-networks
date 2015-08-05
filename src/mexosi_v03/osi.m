function [x,lambda,status] = osi(c,x_min,x_max,A,b_min,b_max,isinteger,Q,options)
% osi Interface to LP/QP solver mexosi
%
% usage: [x,z,status] = osi(c,x_min,x_max,A,b_min,b_max,isinteger,Q,options)
%
% min c'*x + x'*Q*x
% s.t b_min <= A*x <= b_max
%     x_min <=  x  <= x_max
%     x(isinteger) are integers
%
% See osioptions.m for available options
%
% output
%  x      : primal
%  z      : dual
%  status : 1 - optimal, 0 - infeasible, -1 - unbounded

% Original Author Johan Löfberg ETH Zürich (clp.m)
%  Modified by Paul Hines, Sept, 2006

%% repare the inputs
c     = vertical(c);
x_min = vertical(x_min);
x_max = vertical(x_max);
A = sparse(A); % just in case A is not already sparse
b_min = vertical(b_min);
b_max = vertical(b_max);
n = length(c);
m = length(b_min);
[rows,cols] = size(A);
if n~=cols || m~=rows
    error('The constraint matrix A is not sized corretly.');
end

if nargin<7 || isempty(isinteger)
    isinteger = false(n,1);
    isMIP = true;
else
    isinteger = logical(isinteger);
    isMIP = any(isinteger);
end
if nargin<8 || isempty(Q)
    Q = [];
    isQP = false;
else
    isQP = true;
end
if nargin<9 || isempty(options)
    options = osioptions;
end

%% Call the MEX file
[x,lambda,status] = mexosi(n,m,A,x_min,x_max,c,b_min,b_max,isMIP,isQP,isinteger,Q,options);

%% Local functions
%% make a column vector
function x = vertical(x)

[rows,cols] = size(x);

if rows < cols
  x = x.';
end
