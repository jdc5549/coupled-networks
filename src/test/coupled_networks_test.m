function tests = coupled_networks_test
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
import matlab.unittest.fixtures.PathFixture
testCase.applyFixture(matlab.unittest.fixtures.PathFixture('../'));  % subfolders argument doesn't seem to work in 2015b -> 'IncludingSubfolders','true' 
testCase.applyFixture(matlab.unittest.fixtures.PathFixture('../../data/'));
testCase.applyFixture(matlab.unittest.fixtures.PathFixture('../test/'));
testCase.applyFixture(matlab.unittest.fixtures.PathFixture('../mexosi_v03/'));
end

function testNotSmart(testCase)
config_json = '../../config/config_cn_runner_test_not_smart.json';
[~,gc_size,mw_lost] = cn_runner(1, config_json, [], 1);
actSolution = [round(gc_size,4),round(mw_lost)];
expSolution = [0.7587,6791];
verifyEqual(testCase,actSolution,expSolution)
end

function testIdeal(testCase)
config_json = '../../config/config_cn_runner_test_ideal.json';
[~,gc_size,mw_lost] = cn_runner(1, config_json, [], 1);
actSolution = [round(gc_size,4),round(mw_lost)];
expSolution = [0.8393,5341];
verifyEqual(testCase,actSolution,expSolution)
end

function testIntermediate(testCase)
config_json = '../../config/config_cn_runner_test_intermediate.json';
[~,gc_size,mw_lost] = cn_runner(1, config_json, [], 1);
actSolution = [round(gc_size,4),round(mw_lost)];
expSolution = [0.8338,6085];
verifyEqual(testCase,actSolution,expSolution)
end

function testVulnerable(testCase)
config_json = '../../config/config_cn_runner_test_vulnerable.json';
[~,gc_size,mw_lost] = cn_runner(1, config_json, [], 1);
actSolution = [round(gc_size,4),round(mw_lost)];
expSolution = [0.2908,19199];
verifyEqual(testCase,actSolution,expSolution)
end