function [ opts ] = defect_detect_init(class , caffe_path , model)
%% File Path
%change the solver or train_val
if strcmp(model , '1')
     opts.caffe_solver_path = 'models/solver_ZF_segmentation.prototxt';
     opts.caffe_net_path = 'models/defect_test_segmentation.prototxt';
elseif strcmp(model , '2')
     opts.caffe_solver_path = 'models/solver_ZF_detection.prototxt';
     opts.caffe_net_path = 'models/defect_test_detection.prototxt';
end
     opts.caffe_init_weights_path = 'models/proposal_final_ZF.caffemodel';
%      opts.caffe_init_weights_path = fullfile(pwd , 'output' , 'train_models' ,class , 'iter_500');
     mkdir_if_missing(fullfile('output' , 'train_models' , class));
     opts.model_path = fullfile('output' , 'train_models' , class);
%% caffe log path
     mkdir_if_missing(fullfile('output' , 'caffe_log' , class ));
     opts.caffe_init_path = fullfile('output' , 'caffe_log' , class , 'caffe_log');
%% diary path
     timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
      
     mkdir_if_missing(fullfile('output' , 'diary' , class));
     opts.diary_path_loss = fullfile('output' , 'diary' , class , [timestamp , '_loss.txt']);
     opts.fid_loss = fopen( opts.diary_path_loss , 'w+');
     
     opts.diary_path_accurancy_train = fullfile('output' , 'diary' , class ,[timestamp , '_accurancy_train.txt']);
     opts.fid_accurancy_train = fopen( opts.diary_path_accurancy_train , 'w+');
     
     opts.diary_path_accurancy_vali = fullfile('output' , 'diary' , class ,[timestamp , '_accurancy_vali.txt']);
     opts.fid_accurancy_vali = fopen( opts.diary_path_accurancy_vali , 'w+');
%% data path
     opts.data_path = fullfile( 'dataset' , [class , '_def']);
     opts.train_img_path = fullfile(opts.data_path , [class , '_def_img'] , 'train');
     opts.train_data_path = fullfile(opts.data_path , 'train');
     opts.test_data_path = fullfile(opts.data_path , 'test');
%% train parameters
    opts.batch_size = 128; %num of samples to train per batch , reduce it in case of out of gpu memory
    opts.batch_pos = opts.batch_size/4;%num of positive samples
    opts.batch_neg = opts.batch_size/4*3;%num of negative samples
    
    opts.batchSize_hnm = 128;%256   %batch size of hnm ,  reduce it in case of out of gpu memory
    opts.batchAcc_hnm = 4;%
 
    opts.maxiter_all = 30;%30;%num of iteration
    
    opts.do_val = 1;
    opts.batch_size_val = 128;%batch size of validation
    
    if strcmp(class , 'class10')||strcmp(class , 'class7')
         opts.ext = 'bmp';
    else
        opts.ext = 'png';
    end

%% caffe init
    opts.rng_seed=6;
    opts.useGpu = true;
    
    if (opts.useGpu==true)
        opts.gpu_id = 1;%auto_select_gpu;
        opts.caffe_version = 'caffe_master';
        active_caffe_mex(opts.gpu_id , caffe_path);
    else
        caffe.reset_all();
        caffe.set_mode_cpu();
    end    
    
    caffe.init_log(opts.caffe_init_path);
%% caffe solver init
    opts.caffe_solver = caffe.Solver(opts.caffe_solver_path);
    opts.caffe_solver.net.copy_from(opts.caffe_init_weights_path);
    
    input_size =   size(opts.caffe_solver.net.blobs('data').get_data() , 1);
    output_size = size(opts.caffe_solver.net.blobs('labels').get_data() , 1);
    opts.scale = input_size/output_size;
    
%%


end

