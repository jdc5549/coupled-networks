function [grid_to_comm_status] = find_buses_with_power(ps,opt)
% figure out which buses in the system have power to them

C = psconstants;
EPS = 1e-3;

% initialize the ouptut
grid_to_comm_status = ps.bus(:,C.bu.status);
n = length(grid_to_comm_status);

% look at each bus
for i = 1:n
    % if we we are doing 2-way coupling, and this bus has power
    if opt.comm.two_way && grid_to_comm_status(i)
        % figure out which bus this bus is powered from
        sh_index = ps.bus(i,C.bu.power_from_sh);
        % if there has been any load shedding, assume loss of grid_to_comm_status
        if opt.comm.two_way_extreme
            if ps.shunt(sh_index,C.sh.factor)<(1-EPS)
                grid_to_comm_status(i) = false;
            end
        else
            p_failure = 1-ps.shunt(sh_index,C.sh.factor);
            is_failed = rand<p_failure;
            if is_failed
                grid_to_comm_status(i) = false;
            end
        end
    end
end

return

%{ 
OLD STUFF
load_shedding_limit = 0.5; % < This is a big assumption!
n = size(ps.bus,1);
is_powered = true(n,1);
G = ps.bus_i(ps.gen(:,1));
ge_status = ps.gen(:,C.ge.status)==1;
D = ps.bus_i(ps.shunt(:,1));
d_factor = ps.shunt(:,C.sh.factor);
Pg0 = ps.gen(:,C.ge.P).*ge_status;
Pg  = Pg0;
EPS = 1e-6;

% go through each subgrid
grid_list = unique(sub_grids)';
for g = grid_list
    bus_set = find(g==sub_grids);
    % find the amount of generation:
    Gsub = ismember(G,bus_set) & ge_status;
    Pg_sub = sum(Pg(Gsub));
    % find the amount of load shedding:
    Dsub = ismember(D,bus_set);
    mean_load_shedding = mean(1-d_factor(Dsub));
    % if there is no active generation in this subgrid, or there is a lot of load shedding, then no power
    if Pg_sub<EPS || mean_load_shedding > load_shedding_limit
        is_powered(bus_set)=false;
    end
end
%}