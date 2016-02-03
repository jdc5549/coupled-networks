function opt = read_config_json(opt,config_json)
% This function reads a json style config file, and adds the results
% to an options structure

%% Communication system modeling stuff
if exist(config_json,'file')
    config_file = parse_json(fileread(config_json));
    opt.json = config_file{1};

    % Comm model stuff
    opt.comm.python_location = opt.json.python_location;
    opt.comm.comm_model = config_file{1}.comm_model;
    opt.comm.config_json = config_json;
    opt.comm.two_way = opt.json.two_way;
    opt.comm.two_way_extreme = opt.json.two_way_extreme;
    opt.comm.degOfCoupling = config_file{1}.deg_of_coupling;
    opt.comm.pValues = config_file{1}.p_values;
    opt.comm.pMax = config_file{1}.p_max;
    opt.comm.pMin = config_file{1}.p_min;
    opt.comm.qValues = config_file{1}.q_values;
    opt.comm.qMax = config_file{1}.q_max;
    opt.comm.qMin = config_file{1}.q_min;
    opt.comm.networkType = config_file{1}.network_type;
    opt.comm.relpath = find_relpath(opt.comm.comm_model,opt);
    % Debug flag
    opt.debug = (config_file{1}.debug);
    % Simulation settings
    opt.sim.stop_threshold = config_file{1}.grid_gc_threshold;
    opt.sim.stop_on_sep = true;
    opt.sim.use_control = opt.json.use_control; % use info from the comms network
    opt.sim.use_comm_model = opt.json.use_comm_model;
    opt.sim.hpc = config_file{1}.hpc;
    opt.sim.f_over_cost = config_file{1}.lambda;
    opt.optimizer = config_file{1}.optimizer;
    % Verbose flag
    opt.verbose = opt.json.verbose;
else
    error('Could not find config.json file.');
end

end 

function relpath = find_relpath(comm_model, opt)
split_path = strsplit(opt.comm.comm_model,filesep);
relpath = split_path(1:length(split_path)-2);
relpath = sprintf('%s/', relpath{:});    % should give "coupled-networks/" root folder
end

