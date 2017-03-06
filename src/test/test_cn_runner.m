% run this from within coupled-networks/src/

addpath('../src/','../data/','./test/','../config/','./mexosi_v03/')
exp_id = 1;
starting_run = 1;
config_json = '../config/config_cn_runner_test_intermediate.json';

cn_runner(exp_id, config_json, [], starting_run)
