function options = osioptions
% default options for osi.m

%    options.solver           [1 (primal)| 2 (dual), (default 1)].
options.solver = 1;
%    options.maxnumiterations [int>=0 (default 99999999)]
options.maxnumiterations = 99999999;
%    options.maxnumseconds    [int>=0 (default 3600)]
options.maxnumseconds = 3600;
%    options.primaltolerance  [double>=0 (default 1e-7)]
options.primaltolerance = 1e-7;
%    options.dualtolerance    [double>=0 (default 1e-7)]
options.dualtolerance = 1e-7;
%    options.primalpivot      [1 (steepest) | 2 (Dantzig) (default 1)]
options.primalpivot = 1;
%    options.dualpivot        [1 (steepest) | 2 (Dantzig) (default 1)]
options.dualpivot = 1;
%    options.verbose          [0|1|... (default 0)]
options.verbose = 0;
