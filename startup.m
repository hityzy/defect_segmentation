function caffe_path= startup()
%% add path
  curdir =fileparts(mfilename('fullpath'));
  
  addpath(genpath(fullfile(curdir, 'utils')));
  addpath(genpath(fullfile(curdir, 'functions')));
  addpath(genpath(fullfile(curdir, 'detection_train')));
  addpath(genpath(fullfile(curdir, 'detection_test')));
  addpath(genpath(fullfile(curdir, 'analysis_toolkit')));
%   caffe_path = fullfile(curdir, 'external', 'caffe_faster_rcnn', 'matlab');
  caffe_path = fullfile(curdir, 'external', 'caffe_rfcn', 'matlab');
%        caffe_path = fullfile(curdir, 'external', 'caffe_master', 'matlab');
   if exist(caffe_path, 'dir') == 0
        error('matcaffe is missing from external/caffe/matlab');
   end
   
   addpath(genpath(caffe_path));
%%
 mkdir_if_missing(fullfile(curdir, 'output'));


if(isempty(gcp('nocreate')))
    fprintf('Preparing for parpool\n');
    parpool;
end
fprintf('Defect detection startup done\n');

end
