function active_caffe_mex(gpu_id , caffe_path)
 %% set gpu in matlab
    cur_dir = pwd;
    caffe_dir = caffe_path;
    addpath(genpath(caffe_dir));
    cd(caffe_dir);
%% 
    caffe.reset_all();
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id-1);
    cd(cur_dir);
    
    
end
