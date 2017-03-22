function conf  = defect_detect_conf( class , caffe_path )
 %% caffe net init
  conf.useGpu = true;
  if (conf.useGpu==true)
       conf.gpu_id = 1;%auto_select_gpu;
       conf.caffe_version = 'faster_rcnn_external';
       active_caffe_mex(conf.gpu_id , caffe_path);
  else
       caffe.reset_all();
       caffe.set_mode_cpu();
  end 
%% stage1
  conf.net_path1 = 'models/defect_test_segmentation.prototxt';
  conf.weights_path1 = fullfile(pwd , 'output' , 'train_models' , class,  [class , '_segmentation']);
  conf.net1 = caffe.Net(conf.net_path1, 'test');
  conf.net1.copy_from(conf.weights_path1);
  
  conf.pos_thresh = 0.8;
%%  stage2
  conf.net_path2 = 'models/defect_test_detection.prototxt';
  conf.weights_path2 = fullfile(pwd , 'output' , 'train_models' , class,  [class , '_detection']);
  conf.net2 = caffe.Net(conf.net_path2, 'test');
  conf.net2.copy_from(conf.weights_path2);
  %%
  conf.batch_size = 256;
  
    if strcmp(class , 'class10')||strcmp(class , 'class7')
         conf.ext = 'bmp';
    else
         conf.ext = 'png';
    end
  
  conf.pos_thresh2 = 0.3;
  
  conf.overlap_high = 0.2;
  conf.overlap_low = 0.1;
    %% only for test2
    if strcmp(class , 'class8')||strcmp(class , 'class4')||strcmp(class , 'class10')
         conf.patch_size=64;
    else
         conf.patch_size=32;
    end
   conf.stride = conf.patch_size/2;  
end

