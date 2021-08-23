function [GC_size,MW_lost] = cn_runner(bus_outages,coupled_nodes,config_json)
% Inputs:
%  config_json - either a psoptions structure or a json file with
%   options
% bus outages - receive which buses to remove from python

exp_id = 1;
starting_run = '1';
warning off %#ok<*WNOFF>
C = psconstants; % get constants that help us to find the data

%% get/set some options
% this will also read the json file and add that to the options
if ischar(config_json)
    opt = psoptions;
    opt = read_config_json(opt,config_json);
else 
    error('No config.json file found.');
end
opt.sim.pid = num2str(feature('getpid'));
pid = opt.sim.pid;
original_exp_id = exp_id;
if ischar(starting_run)
    starting_run = str2num(starting_run);
end

if isdeployed
    exp_id = str2num(exp_id); % Do this otherwise exp_id is treated as ASCII
end
exp_id = exp_id + starting_run;
if strcmp(opt.sim.hpc,'HIVE') % When run on the HIVE with condor the exp_id starts at 0
    exp_id = exp_id + 1;
end
rng(exp_id) %Set the seed to the experiment number to get consistent random draws each time that run
exp_id = num2str(exp_id);

%% Load the data
if ~isdeployed && opt.verbose
    fprintf('----------------------------------------------------------\n');
%     fprintf('loading the data, sep_threshold = %f\n', opt.sim.sep_threshold);
    fprintf('loading the data, stop_threshold = %f\n', opt.sim.stop_threshold);
    tic
end

if exist(opt.json.power_systems_data_location,'file')
    file = opt.json.power_systems_data_location;
    %[~,ext] = strtok(opt.ps_file_location,'.');
    if regexp(file,'\.mat$')
        load(opt.json.power_systems_data_location);
    elseif strcmp(ext,'\.m$')
        ps = feval(opt.json.power_systems_data_location);
        ps = updateps(ps);
    else
        error('Unknown file type');
    end 
else
    error('No power systems data provided. Set powerSystemsDataLocation in config.json');
end
% Pre-process the data
n = size(ps.bus,1);
Pd0 = ps.shunt(:,C.sh.P).*ps.shunt(:,C.sh.factor);
Pd0_sum = sum(Pd0);

%% Get the coupled nodes and write them to file
coupled_nodes_filename = strcat('/tmp/', 'coupled_nodes_', pid, '.csv');
% sprintf('Single coupled nodes file name: %s', coupled_nodes_filename)

% Write the header
fileID = fopen(coupled_nodes_filename,'w');
fprintf(fileID,'%s\r\n','node');
fclose(fileID);

% Write the data
dlmwrite(coupled_nodes_filename, coupled_nodes,'-append','delimiter',',','roffset', 0,'coffset',0);


%% Write the grid status to the ps structure
% prepare and write the grid status
grid_status = ones(n,1);
grid_status(bus_outages) = 0;
ps.bus(:,C.bu.status) = grid_status;

%% Write information about the comm system
if opt.sim.use_comm_model
    grid_comm_connectivity = zeros(n,1);
    grid_comm_connectivity(coupled_nodes) = 1;

    ps.bus(:,C.bu.grid_comm) = grid_comm_connectivity;
    % Write an initial comm status file
    comm_status_file = sprintf('/tmp/comm_status_%s.csv',pid); % read from comms model
    f = fopen(comm_status_file,'w');
    fprintf(f,'node,status\n');
    fprintf(f,'%d,1\n',(1:n));
    fclose(f);
end

[is_blackout,~,MW_lost,p_out] = dcsimsep(ps,[],bus_outages,opt);
GC_size = 1 - p_out;

function coupled_nodes = get_coupled_nodes_from_file(exp_id, q_point,opt,n)
%%q is defined in the opposite direction of p so at a q of 0.99 we want to return a list of 99% of 
%the nodes in a network. This is the reason for taking 1-q_point.

q_point = 1-q_point;  
numOutagesInCouplingFile = 100; % TODO get this in opt.comm
inverseStepSizeForCoupling = 100; % TODO get this in opt.comm
minCouplingPoint = 0.01; % TODO get this in opt.comm

% split_path = strsplit(opt.comm.comm_model,filesep);
% relpath = split_path(1:length(split_path)-2);
% relpath = sprintf('%s/', relpath{:});    % should give "coupled-networks/" root folder

comm_file_name = [opt.comm.relpath,'data/coupled-nodes/coupled_node_',sprintf('%s',exp_id),'.csv.bz2']; % %s so exp_id gets converted to a string
[~,name,~] = fileparts(comm_file_statusname);
system_call = ['bzip2 -ckd ',comm_file_name,' > ','/tmp/',name];
system(system_call); % unzip the file and put it in /tmp/
comm_file_name = ['/tmp/',name];
fprintf('Coupled nodes file name: %s\n', comm_file_name);
coupled_nodes = csvread(comm_file_name,1,0);

p_calc = numOutagesInCouplingFile - inverseStepSizeForCoupling*(q_point-minCouplingPoint);
p_column = round(p_calc) - 1;  % 100-100*(0.99-0.01) - 1 = 1, round first otherwise you can get unexpected results
% print "p_column " + str(p_column) + " from file: " + file_name
if p_column > numOutagesInCouplingFile || p_column == 0 || n ~= 2383
    randomRemovalFraction=1-q_point;  % Fraction of nodes to remove
    numcoupled_nodes = floor(randomRemovalFraction * n);
    coupled_nodes = randsample(n,numcoupled_nodes);
    if opt.verbose == true
        fprintf('No outages found for q_point %f at column %f for %f nodes, creating outage instead\n',(1-q_point),p_column,n);
        fprintf('Size coupled_nodes list %d for run %d at q_point %f\n',length(coupled_nodes),run,(1-q_point));
    end
else
    try
        coupled_nodes = coupled_nodes(:,p_column);
        coupled_nodes = coupled_nodes(coupled_nodes ~= 0);
        num_coupled_nodes = length(coupled_nodes);
        if opt.verbose == true
            fprintf('Size coupled_nodes list %d for run %s, q_point %f, p_column %d\n',num_coupled_nodes,exp_id,1-q_point,p_column);
        end
    catch
        error('Coupled node outage read error')
    end
end