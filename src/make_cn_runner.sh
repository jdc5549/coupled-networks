#!/bin/bash
#ROOT_DIR=/projects/CN/coupled-networks/source/compiled-matlab/
ROOT_DIR=~/workspace/coupled-networks/source/compiled-matlab/

mkdir ${ROOT_DIR}cn_runner
mkdir ${ROOT_DIR}cn_runner/distrib
mkdir ${ROOT_DIR}cn_runner/src

mcc -R -nodisplay -o cn_runner -W main:cn_runner -T link:exe -d ${ROOT_DIR}cn_runner/src -w enable:specified_file_mismatch -w enable:repeated_file -w enable:switch_ignored -w enable:missing_lib_sentinel -w enable:demo_license -v ${ROOT_DIR}cn_runner.m -a ${ROOT_DIR}check_separation.m -a ${ROOT_DIR}dcpf.m -a ${ROOT_DIR}dcsimsep.m -a ${ROOT_DIR}emergency_control.m -a ${ROOT_DIR}findSubGraphs.m -a ${ROOT_DIR}find_buses_with_power.m -a ${ROOT_DIR}mexosi_v03/mexosi.m -a ${ROOT_DIR}mexosi_v03/mexosi.mexa64 -a ${ROOT_DIR}mexosi_v03/osi.m -a ${ROOT_DIR}mexosi_v03/osioptions.m -a ${ROOT_DIR}parse_json.m -a ${ROOT_DIR}psconstants.m -a ${ROOT_DIR}psoptions.m -a ${ROOT_DIR}read_config_json.m -a ${ROOT_DIR}redispatch.m -a ${ROOT_DIR}relay_settings.m -a ${ROOT_DIR}take_control_actions.m -a ${ROOT_DIR}total_P_mismatch.m -a ${ROOT_DIR}update_relays.m -a ${ROOT_DIR}updateps.m
