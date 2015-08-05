function ps = take_control_actions(ps,sub_grids,ramp_rate,dt,it_no,opt)
% usage: ps = take_control_actions(ps,sub_grids,ramp_rate,dt,it_no,opt)
%
% Compute and implement emergency control actions.
%  Interfaces with comm model, if requested in the options

% Constants
C = psconstants;
EPS = 1e-3;
% Collect some data from the system
n = size(ps.bus,1);
m = size(ps.branch,1);
F = ps.bus_i(ps.branch(:,C.br.from));
T = ps.bus_i(ps.branch(:,C.br.to));
flow = ps.branch(:,C.br.Pf);
flow_max = ps.branch(:,C.br.rateB);
ge_status = ps.gen(:,C.ge.status);
Pg_max = ps.gen(:,C.ge.Pmax).*ge_status + EPS;
Pg_min = ps.gen(:,C.ge.Pmin).*ge_status - EPS;
G = ps.bus_i(ps.gen(:,1));
D = ps.bus_i(ps.shunt(:,1));
comm_failed_set = [];

% Check the mismatch before we start, just as a debug step
mis_old = total_P_mismatch(ps);
if abs(mis_old)>EPS
    error('System not balanced on entry to take_control_actions');
end

% If we are to use the comm model do:
if opt.sim.use_comm_model
    pid = opt.sim.pid;
    if ispc
        comm_status_file = sprintf('C:\Temp\comm_status_%s.csv',pid);
        grid_status_file = sprintf('C:\Temp\grid_status_%s.csv',pid);
    else
        comm_status_file = sprintf('/tmp/comm_status_%s.csv',pid); % read from comms model
        grid_status_file = sprintf('/tmp/grid_status_%s.csv',pid); % written by grid model
    end
    % get the level of coupling between the networks
    grid_comm = ps.bus(:,C.bu.grid_comm);

    % For the one-way model, the comm status is un-changed, therefore:
    if opt.comm.two_way
        % Write out the status of each node in the network
        station_service_status = find_buses_with_power(ps,opt);
        %ps.bus(:,C.bu.status) = staTrition_service_status;
        %  output file name is grid_status_{pid}.csv
        % The comm system is only affected if there is a bi-directional link
        % grid_to_comm_status = station_service_status | (~ps.bus(:,C.bu.grid_comm));
        % The comm system knows which nodes are coupled so pass in all outages, 
        % couplednetworks.py will figure out which ones are really out.
        grid_to_comm_status = station_service_status | ~grid_comm;
    else
        grid_to_comm_status = ones(n,1);
    end
    % Write a header to the file
    fileID = fopen(grid_status_file,'w');
    fprintf(fileID,'bus,status\n');
    fclose(fileID);
    % Write the data
    dlmwrite(grid_status_file, [ps.bus(:,1) grid_to_comm_status],'-append','delimiter',',');
    % Call python comms code
    % if ~all(grid_to_comm_status)
    % only call the comms model if we're doing two_way coupling
    if opt.comm.two_way || opt.comm.two_way_extreme
        % Figure out where the python code is
        python_location = opt.comm.python_location;
        comm_model = opt.comm.comm_model;
        json_file = opt.comm.config_json;
        % if it exists, call the comm model code
        if exist(comm_model,'file')
            systemcall = sprintf('%s %s %s %d %d %s',python_location,comm_model,pid,it_no,-1,json_file);
            system(systemcall);
        end
    end
    % check the status of the comm network
    % read the file /tmp/comm_status_{pid}.csv
    % bus,status
    % 1,1
    % 2,0 etc...
    if exist(comm_status_file,'file')
        data = csvread(comm_status_file,1);
        if size(data,1)~=n
            error('Wrong number of items in comm status file');
        end
        is_comm_node_operational = (data(:,2)==1);
        comm_failed_set = find(~is_comm_node_operational);
        % pct_failed = (1-mean(is_comm_node_operational))*100;
        %fprintf('%.2f %% of comm nodes have failed\n',pct_failed);
        % Comm status is the combination of the two things:
        comm_status = grid_comm & is_comm_node_operational;
        % delete(comm_status_file); % this is now deleted by the calling process. delete the file to avoid filling up /tmp
        %keyboard
    else
        disp('Warning: Could not read comm status file');
        comm_status = ps.bus(:,C.bu.comm_status);
        comm_failed_set = [];
        % is_comm_node_operational = true(n,1);
    end
    % Read power system data (flow) from each operating node
    is_br_readable = comm_status(F) | comm_status(T);
    measured_flow = nan(m,1);
    measured_flow(is_br_readable) = flow(is_br_readable);
    measured_branch_st = true(m,1);
    measured_branch_st(is_br_readable) = ps.branch(is_br_readable,C.br.status);
else
    measured_flow = flow;
    measured_branch_st = ps.branch(:,C.br.status);
    comm_status = true(n,1);
end

% if there are overloads in the system, try to mitigate them
if any(abs(measured_flow)>flow_max)
    % Figure out the ramp rate
    ramp_dt = min(dt,opt.sim.dt_max_default); % the amount of generator ramping time to allow
    max_ramp = ramp_rate*ramp_dt;
    % Find the optimal load/gen shedding
    [delta_Pd,delta_Pg] = emergency_control(ps,measured_flow,measured_branch_st,max_ramp,comm_status,opt);
    % If emergency control says that we should do something:
    if any(abs(delta_Pd)>EPS)
        % check to see which loads/gens can be controlled 
        %  (These lines shouldn't be needed)
        is_D_failed = ismember(D,comm_failed_set);
        delta_Pd(is_D_failed) = 0;
        is_G_failed = ismember(G,comm_failed_set);
        delta_Pg(is_G_failed) = 0;
        % Compute the new amount of generation
        Pg_new = ps.gen(:,C.ge.P).*ps.gen(:,C.ge.status) + delta_Pg;
        % Error checking
%        if any( Pg_new<Pg_min | Pg_new>Pg_max )
        if any( round(Pg_new)<round(Pg_min-EPS) | round(Pg_new)>round(Pg_max+EPS) )
            error('Pg_new is out of bounds');
        end
        % Compute the new load factor
        delta_lf = delta_Pd./ps.shunt(:,C.sh.P);
        delta_lf(isnan(delta_lf)) = 0;
        lf_new = ps.shunt(:,C.sh.factor) + delta_lf;
        % Implement the results
        ps.gen(:,C.ge.P) = max(Pg_min,min(Pg_new,Pg_max)); % implement Pg
        ps.shunt(:,C.sh.factor) = max(0,min(lf_new,1));
    end
    % For extreme two-way case, trip generators if there is a comm failure
    if opt.sim.use_comm_model && opt.comm.two_way_extreme
        is_gen_failed = ismember(G,comm_failed_set);
        if opt.verbose
            ge_st = (ps.gen(:,C.ge.status)==1);
            new_failure = (is_gen_failed & ge_st);
            n_new_failures = sum(new_failure);
            if n_new_failures>0
                fprintf('Tripping %d generators due to comm failures.\n',n_new_failures);
            end
        end
        ps.gen(is_gen_failed,C.ge.P) = 0;
        ps.gen(is_gen_failed,C.ge.status) = 0;
    end
end

% Get the new mismatch
mis_new = total_P_mismatch(ps,sub_grids);
% If there was an error in the balance, run redispatch again
if abs(mis_new)>EPS
    ps = redispatch(ps,sub_grids,max_ramp,opt.verbose);
    ge_status = ps.gen(:,C.ge.status);
    Pg_max = ps.gen(:,C.ge.Pmax).*ge_status + EPS;
    Pg_min = ps.gen(:,C.ge.Pmin).*ge_status - EPS;
end
% Run dcpf to get power flows
ps = dcpf(ps,sub_grids,true,opt.verbose);
% Check to make sure things are within bounds
Pg = ps.gen(:,C.ge.Pg);
%         if any( Pg>Pg_max | Pg<Pg_min )
if any( round(Pg)<round(Pg_min-EPS) | round(Pg)>round(Pg_max+EPS) )
    error('Pg is out of bounds');
end

