function [ps_py,t,done,p_out] = cn_step_def(delta_Pd,delta_Pg,bus_outages,t,it_no,ps,opt)  
% is_blackout indicates whether a large separation occurs
% branches_lost gives the set of dependant outages that occur due to relay actions
%  this is a ? by 2 matrix, with t in the first column, br no. in the second
% bus_outages gives the bus indices associated with bus failures
% MW_lost indicates how much load was lost as a result of small separations
% p_out is proportion of buses separated  
% busessep is a list of the buses that separated 
% flows is the power flow across each branch at each timestep
% times is the time at each timestep
% power_flow is the power flow at each timestep

%transpose the deltas
delta_Pd = delta_Pd.';
delta_Pg = delta_Pg.';

% init the outputs
p_out=0;  
busessep=[];  

% Grab some useful data
C = psconstants;
EPS = 1e-4;
dt_max = opt.sim.dt_max_default;
t_max = 60*30; % time limit for the simulation
n = size(ps.bus,1);
m = size(ps.branch,1);
F = ps.bus_i(ps.branch(:,1));
T = ps.bus_i(ps.branch(:,2));
G = ps.bus_i(ps.gen(:,1));   % gen index
ge_status = ps.gen(:,C.ge.status);
Pg_max = ps.gen(:,C.ge.Pmax).*ge_status + EPS;
Pg_min = ps.gen(:,C.ge.Pmin).*ge_status - EPS;
Pg = ps.gen(:,C.ge.Pg).*ge_status;
D = ps.bus_i(ps.shunt(:,1)); % load index
flow = ps.branch(:,C.br.Pf);
flow_max = ps.branch(:,C.br.rateB);

%NO_SEP = 0;
BIG_SEP = 2;
%SMALL_SEP = 1;
% set the power plant ramp rates
ramp_rate = ps.gen(:,C.ge.ramp_rate_up)/60; % ramp rate in MW/second
if all(ramp_rate==0)
    ramp_rate_MW_per_min = max(1,Pg_max*.05); % assume that all plants can ramp at 5% per minute. 
                                            % for a 100 MW plant, this
                                            % would be 5 MW/min. Reasonable
    ramp_rate = ramp_rate_MW_per_min/60;
end

if ~isempty(bus_outages) %means we are at time t = 0
    Pd0 = ps.shunt(:,C.sh.P).*ps.shunt(:,C.sh.factor);
    Pd0_sum = sum(Pd0);
    Pg0_sum = sum(Pg);
    relay_outages = zeros(0,2);
    ps.relay = relay_settings(ps,false,true);

    % Step 1. redispatch and run the DCPF
    if ~isfield(ps,'bus_i')
        ps = updateps(ps);
    end
    br_st = ps.branch(:,C.br.status)~=0;

    % check to make sure that the base case is load balanced
    if opt.debug && abs(Pd0_sum - Pg0_sum)>EPS
        error('The base case power system is not load balanced');
    end
    [sub_grids,n_sub_old] = findSubGraphs(ps.bus(:,1),ps.branch(br_st,1:2));
    if n_sub_old>1
        error('The base case has more than one island');
    end
    % Find the ramp rate
    ramp_rate( ~ge_status ) = 0; % plants that are shut down cannot ramp
    % Check the mismatch
    mis = total_P_mismatch(ps);
    if opt.debug && abs(mis)>EPS, error('Base case has mismatch'); end
    % Calculate the power flow
    ps = dcpf(ps,[],false,opt.verbose); % this one should not need to do any redispatch, just line flow calcs
    % Get the power flow
    flow = ps.branch(:,C.br.Pf);

    t = 1;
    % Error check
    Pg = ps.gen(:,C.ge.Pg);
    % if opt.debug && any( Pg<Pg_min | Pg>Pg_max )
    if opt.debug && any( Pg>Pg_max )
        error('Pg is out of bounds');
    end

    % Step 2. Apply exogenous outages
    if opt.verbose
        fprintf('------- t = %.3f ----------\n',t);
        fprintf('Exogenous events:\n');
    end
    % Apply the bus outages
    for i=1:length(bus_outages)
        bus_no = bus_outages(i);
        bus_ix = ps.bus_i(bus_no);
        if opt.debug && isempty(bus_ix) || bus_ix<=0 || bus_ix>n
            error('%d is not a valid bus number',bus_no);
        end
        br_set = (F==bus_ix) | (T==bus_ix);
        ps.branch(br_set,C.br.status) = 0;
        % trip gens and shunts at this bus
        ps.gen  (G==bus_ix,C.ge.status) = 0;
        ps.shunt(D==bus_ix,C.sh.status) = 0;
        if opt.verbose
            fprintf(' Removed bus %d\n',bus_no);
        end
    end
end

dt = 10.00; % This initial time step sets the quantity of initial gen ramping.
% Step 3. Find sub-grids in the network and check for major separation
[sep,sub_grids,n_sub,p_out,busessep] = check_separation(ps,opt.sim.stop_threshold,opt.verbose);


% Step 4. redispatch & run the power flow
% if there are new islands, redispatch the generators
if n_sub>1
    ramp_dt = max(dt,opt.sim.fast_ramp_mins*60); % the amount of 
       % generator ramping time to allow. 
    max_ramp = ramp_rate*ramp_dt; 
    [Pg,ge_status,d_factor] = redispatch(ps,sub_grids,max_ramp,opt.verbose);
    % Error check:
    Pg_max = ps.gen(:,C.ge.Pmax).*ge_status + EPS;
    Pg_min = ps.gen(:,C.ge.Pmin).*ge_status - EPS;
%         if opt.debug && any( Pg<Pg_min | Pg>Pg_max ), error('Pg is out of bounds'); end
%         if any( round(Pg)<round(Pg_min-EPS) | round(Pg)>round(Pg_max+EPS) ), error('Pg is out of bounds'); end
    if any( round(Pg)>round(Pg_max+EPS) ), error('Pg is out of bounds'); end
    % Implement the changes to load and generation
    ps.shunt(:,C.sh.factor) = d_factor;
    ps.gen(:,C.ge.status) = ge_status;
    ps.gen(:,C.ge.P) = Pg;
    ramp_rate(~ge_status)=0; % make sure that failed generators don't ramp
end

n_sub_old = n_sub;
% run the power flow and record the flow
ps = dcpf(ps,sub_grids,true,opt.verbose);
if opt.debug
    % Check that Pg is within bounds
    ge_status = ps.gen(:,C.ge.status);
    Pg_max = ps.gen(:,C.ge.Pmax).*ge_status + EPS;
    Pg_min = ps.gen(:,C.ge.Pmin).*ge_status - EPS;
    Pg = ps.gen(:,C.ge.Pg);
    if any( round(Pg)>round(Pg_max+EPS) ) 
        error('Pg is out of bounds');
    end        
end

% Extract and record the flows
flow  = ps.branch(:,C.br.Pf);
Pd = ps.shunt(:,C.sh.P).*ps.shunt(:,C.sh.factor);

% Step 4a. Take control actions from RL agent
if opt.sim.use_control
    comm_failed_set = [];

    % Check the mismatch before we start, just as a debug step
    mis_old = total_P_mismatch(ps);
    if abs(mis_old)>EPS
        disp('System not balanced on entry to take_control_actions');
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
        %Rescale deltas into range of Pg and Pd
        delta_Pd = delta_Pd.*Pd / 2;
        for i=1:length(Pg)
            delta_Pg(i) = delta_Pg(i)*(Pg(i)-Pg_min(i))/2;
        end
        % If RL agent says that we should do something:
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
end

% Step 5. Update relays
[ps.relay,br_out_new,dt,n_over] = update_relays(ps,opt.verbose,dt_max);
if opt.verbose && n_over>0
    fprintf(' There are %d overloads in the system\n',n_over);
end

% advance/print the time
t = t + dt;
if t >= t_max
    done = true;
    % do a final redispatch just to make sure
    [Pg,ge_status,d_factor] = redispatch(ps,sub_grids,ramp_rate*dt,opt.verbose);
    % Error check
    % if opt.debug
    Pg_max = ps.gen(:,C.ge.Pmax).*ge_status + EPS;
    Pg_min = ps.gen(:,C.ge.Pmin).*ge_status - EPS;
    %     if any( Pg<Pg_min | Pg>Pg_max ), error('Pg is out of bounds'); end
    if any( round(Pg)<round(Pg_min-EPS) | round(Pg)>round(Pg_max+EPS) ), error('Pg is out of bounds'); end
    % end
    % Implement
    ps.shunt(:,C.sh.factor) = d_factor;
    ps.gen(:,C.ge.status) = ge_status;
    ps.gen(:,C.ge.P) = Pg;
    %}
    % Compute the amount of load lost
    Pd = ps.shunt(:,C.sh.P).*ps.shunt(:,C.sh.factor);
    n_GC = n - length(busessep);
else
    done = false;
end
% Step 6. Trip overloaded branches
ps.branch(br_out_new,C.br.status) = 0;

%convert ps sparse matrices into full matrices so they can be returned to python
ps_py = ps;
ps_py.bus_i = full(ps_py.bus_i);
ps_py = rmfield(ps_py,'B');