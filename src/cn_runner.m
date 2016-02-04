% function cn_runner(exp_id, config_json)
function [is_blackout,GC_size,MW_lost] = cn_runner(exp_id, config_json, exps_to_run_file, starting_run)
% Inputs:
%  exp_id - experiment ID
%  config_json - either a psoptions structure or a json file with
%   options
%  exps_to_run - headerless csv file with list of experiment numbers to run
%  starting_run - run number to start with for multi-submission experiments
%   
% Copyright 2016 The MITRE Corporation and The University of Vermont

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

if exist(exps_to_run_file,'file')  % For use in re-doing runs
    if isdeployed
        exp_id = str2num(exp_id); 
    end
    exp_id = exp_id + starting_run;
    if strcmp(opt.sim.hpc,'HIVE') % When run on the HIVE with condor the exp_id starts at 0
        exp_id = exp_id + 1; % Here exp_id is the index into exps_to_run which can't be 0
    end
    exps_to_run = csvread(exps_to_run_file);    % Expects no header 
    exp_id = exps_to_run(exp_id); % Get the actual exp. from the index
    if strcmp(opt.sim.hpc,'HIVE')
        exp_id = exp_id + 1;  % if we made the expt_to_run_file from a run on the HIVE then we'll need to add 1 to get the right coupled_nodes and bus_outages
    end
    rng(exp_id) %Set the seed to the experiment number to get consistent random draws each time that run
    exp_id = num2str(exp_id);
    disp(['Running experiment number ',num2str(exp_id),' from file: ',exps_to_run_file]);
else
    if isdeployed
        exp_id = str2num(exp_id); % Do this otherwise exp_id is treated as ASCII
    end
    exp_id = exp_id + starting_run;
    if strcmp(opt.sim.hpc,'HIVE') % When run on the HIVE with condor the exp_id starts at 0
        exp_id = exp_id + 1;
    end
    rng(exp_id) %Set the seed to the experiment number to get consistent random draws each time that run
    exp_id = num2str(exp_id);
    disp(['Running experiment number ',num2str(exp_id)]);
end



if isdeployed
    opt.verbose = false;
end

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
coupled_nodes = get_coupled_nodes_from_file(exp_id,opt.comm.degOfCoupling,opt,n);
coupled_nodes_filename = strcat('/tmp/', 'coupled_nodes_', pid, '.csv');
% sprintf('Single coupled nodes file name: %s', coupled_nodes_filename)

% Write the header
fileID = fopen(coupled_nodes_filename,'w');
fprintf(fileID,'%s\r\n','node');
fclose(fileID);

% Write the data
dlmwrite(coupled_nodes_filename, coupled_nodes,'-append','delimiter',',','roffset', 0,'coffset',0);

if ~isdeployed && opt.verbose
    toc
    fprintf('----------------------------------------------------------\n');
end

if isempty(coupled_nodes)
    coupled_nodes = [];
    if ~isdeployed && opt.verbose
        disp('No coupled nodes specified.');
    end
end


%% Get some stats
if strcmp(opt.sim.hpc,'HIVE') && isdeployed
    hive_node = '';
    try
        proc_id = str2num(original_exp_id);
        hive_node = sprintf('condor_q -l -submitter veneman -const ''ProcId == %d'' | grep RemoteHost',proc_id);
        fprintf('Getting RemoteHost info with call: %s\n',hive_node);
        [~,cmdout] = system(hive_node);
        fprintf('RemoteHost info: %s\n',cmdout);
    catch
        fprintf(2,'Attempt to get remote host information failed with call: %s\n', hive_node);
    end
end
% start = posixtime(datetime('now'));  % doesn't work in R2013a
start = now;

%% Loop through all the different outages
p_values = opt.comm.pValues;
p_min = opt.comm.pMin;
p_range = opt.comm.pMax - p_min;
results = zeros(p_values,4);

for i = 0:p_values
    tic;
    p_point = i/p_values;
    p_point = p_point/(1/p_range)+p_min;
    
    if p_point == 0.0
        results(i+1,:) = [p_point,0,0.0,Pd0_sum];  % if p=0 you're going to remove all nodes and you know it will fail so don't bother simulating
        continue
    end
    if p_point == 1.0
        results(i+1,:) = [p_point,1,1.0,0.0];  % if p=1 you won't remove any nodes and success is guaranteed so don't bother simulating
        continue
    end
    
    randomRemovalFraction=1-p_point;  % Fraction of nodes to remove
    %print "randomRemovalFraction " + str(randomRemovalFraction)
    numNodesAttacked = floor(randomRemovalFraction * n);
    
    bus_outages = get_bus_outages_from_file(exp_id,p_point,opt,n);

    % comm_status is written by couplednetworks.py and is unknown at this point so it shouldn't be written
    % grid_status is written in take_control_actions which also checks that a bus has power before writing it

    if isempty(bus_outages)
        bus_outages = [];
        if ~isdeployed && opt.verbose
            disp('No bus outages specified.');
        end
    end

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

    %% run the simulator
    if isdeployed
        try 
            [is_blackout,~,MW_lost,p_out] = dcsimsep(ps,[],bus_outages,opt);
            GC_size = 1 - p_out;
        catch  %#ok<CTCH>
            fprintf(2,'Warning: Error during dcsimsep run. Trying again...\n');
            try
                [is_blackout,~,MW_lost,p_out] = dcsimsep(ps,[],bus_outages,opt);
                GC_size = 1 - p_out;
            catch
                fprintf(2,'Warning: Error during dcsimsep run on 2nd try.\n');
                is_blackout=NaN;
                GC_size=NaN;
                MW_lost=NaN;
                output_folder = strcat(opt.comm.relpath, 'output/', opt.comm.networkType, '/');
                if ~exist(output_folder,'dir')
                    mkdir(output_folder);
                end
                error_folder = strcat(output_folder,'/error/');
                if ~exist(error_folder,'dir')
                    mkdir(error_folder);
                end
                error_file_name = sprintf('%s/error_case_data_exp_%s_p_point_%s_time_%s.mat',error_folder,exp_id,num2str(p_point),datestr(now,30));
                try
                    save(error_file_name,'p_point','exp_id','coupled_nodes','grid_status','bus_outages','i','opt','ps','original_exp_id');
                    fprintf(2,'Wrote error to file at %s\n',error_file_name);
                catch
                    fprintf(2,'Error writing to error file at %s\n',error_file_name);
                end
            end
        end
    else
        [is_blackout,~,MW_lost,p_out] = dcsimsep(ps,[],bus_outages,opt);
        GC_size = 1 - p_out;
    end
    
    results(i+1,:) = [p_point,is_blackout,GC_size,MW_lost];

    if ~isdeployed && opt.verbose
        if is_blackout
            disp('Blackout');
        else
            disp('Not blackout');
        end
    end

    fprintf('Run time for outage point %.3f in experiment %s was %.2fs\n',p_point,exp_id,toc);

end % end outage loop

%% Write the output file
output_folder = strcat(opt.comm.relpath, 'output/', opt.comm.networkType, '/');
if ~exist(output_folder,'dir')
    mkdir(output_folder);
end
copyfile(config_json,output_folder);
output_filename = strcat(output_folder, opt.comm.networkType, '_', exp_id, '.csv');
if exist(output_filename, 'file') == 2 % if the file exists give it a different name before writing it
    output_filename = strcat(output_folder, opt.comm.networkType, '_', exp_id, '_', datestr(now,'ddmmmyyyy_HHMMSS'),'.csv');
end

% Write the header
fileID = fopen(output_filename,'w');
fprintf(fileID,'%s\r\n','p,Pinf,GC_size,MW_lost');
fclose(fileID);

% Write the data
dlmwrite(output_filename,results,'-append','delimiter',',','roffset', 0,'coffset',0);

%% Cleanup temp files
delete(strcat('/tmp/','*',pid,'.csv'));
delete(strcat('/tmp/','coupled_node_',exp_id,'.csv'));
delete(strcat('/tmp/','bus_outage_',exp_id,'.csv'));

% run_time = posixtime(datetime('now')) - start;   % doesn't work in R2013a 
run_time = (now-start)*datenum(1)*86400;
fprintf('Total run time for experiment %s was %.2fs',exp_id,run_time); 

end % end cn_runner function

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
[~,name,~] = fileparts(comm_file_name);
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

end % end coupled nodes function

function bus_outages = get_bus_outages_from_file(exp_id, p_point,opt,n)
%%q is defined in the opposite direction of p so at a q of 0.99 we want to return a list of 99% of 
%the nodes in a network. This is the reason for taking 1-q_point.

numOutagesInFile = 100; % TODO get this in opt.comm
inverseStepSize = 200; % TODO get this in opt.comm
minOutagePoint = 0.5; % TODO get this in opt.comm

% split_path = strsplit(opt.comm.comm_model,filesep);
% relpath = split_path(1:length(split_path)-2);
% relpath = sprintf('%s/', relpath{:});    % should give "coupled-networks/" root folder

outage_file_name = [opt.comm.relpath,'data/node-removals/bus_outage_',sprintf('%s',exp_id),'.csv.bz2']; % %s so exp_id gets converted to a string
[~,name,~] = fileparts(outage_file_name);
system_call = ['bzip2 -ckd ',outage_file_name,' > ','/tmp/',name];
system(system_call); % unzip the file and put it in /tmp/
outage_file_name = ['/tmp/',name];
% fprintf('Outage file name: %s', outage_file_name);
bus_outages = csvread(outage_file_name,1,0);

p_calc = numOutagesInFile - inverseStepSize*(p_point-minOutagePoint);
p_column = round(p_calc);  % 50-10000*(0.995-0.75)/50 = 1, round first otherwise you can get unexpected results

% print "p_column " + str(p_column) + " from file: " + file_name
if p_column > numOutagesInFile || p_column == 0 || n ~= 2383
    randomRemovalFraction=1-p_point;  % Fraction of nodes to remove
    num_bus_outages = floor(randomRemovalFraction * n);
    bus_outages = randsample(n,num_bus_outages);
    if opt.verbose == true
        fprintf('No outages found for q_point %f at column %d for %d nodes, creating outage instead.\n',p_point,p_column,n);
        fprintf('Size bus_outages list %d for run %d, p_point %f.\n',length(bus_outages),exp_id,p_point);
    end
else
    try
        bus_outages = bus_outages(:,p_column);
        bus_outages = bus_outages(bus_outages ~= 0);
        num_bus_outages = length(bus_outages);
        if opt.verbose == true
            fprintf('size bus_outages list %d for run %d, p_point %f, p_column %d\n',num_bus_outages,exp_id,p_point,p_column);
        end
    catch
        error('Bus outage read error')
    end
end

end % end bus outages function