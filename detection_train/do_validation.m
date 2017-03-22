function [ accurancy_fg , accurancy_bg ] = do_validation( opts , pos_data_test , neg_data_test )
%% using test date to do validation
%% do validation fg
    n_fg = size(pos_data_test,4);
    nBatches = ceil(n_fg/opts.batch_size_val);
    result=0;
    for i=1:nBatches

        batch = pos_data_test(:,:,:,opts.batch_size_val*(i-1)+1:min(end,opts.batch_size_val*i));
        % permute data into caffe c++ memory, thus [num, channels, height, width]
        batch = batch(:, :, [3, 2, 1], :); % from rgb to brg
        batch = permute(batch, [2, 1, 3, 4]);
        batch = single(batch);
        label_tmp = rand( size(batch , 1)/opts.scale , size(batch , 2)/opts.scale , 1 , size(batch , 4) ,'single');
        
        % Reshape net's input blobs
        opts.caffe_solver.net.blobs('data').reshape(size(batch)); % reshape blob 'data'
        opts.caffe_solver.net.blobs('labels').reshape(size(label_tmp));
        opts.caffe_solver.net.reshape();
        
        opts.caffe_solver.net.blobs('data').set_data(batch);
        opts.caffe_solver.net.blobs('labels').set_data(label_tmp);
        opts.caffe_solver.net.forward_prefilled();
   
        prob = opts.caffe_solver.net.blobs('proposal_cls_prob').get_data();
        score_fg = prob(:,:,1,:);
        score_fg = squeeze(sum(sum(score_fg ,1) , 2)) /(size(label_tmp,1)*size(label_tmp,2));
        
        score_bg = prob(:,:,2,:);
        score_bg = squeeze(sum(sum(score_bg ,1) , 2)) / (size(label_tmp,1)*size(label_tmp,2));
    
        res_fg = score_fg>score_bg;
        result = result+sum(res_fg);
    end
         accurancy_fg = result/n_fg;
%%

%% do validation bg
    n_bg = size(neg_data_test,4);
    nBatches = ceil(n_bg/opts.batch_size_val);
    result=0;
    for i=1:nBatches

        batch = neg_data_test(:,:,:,opts.batch_size_val*(i-1)+1:min(end,opts.batch_size_val*i));
        % permute data into caffe c++ memory, thus [num, channels, height, width]
        batch = batch(:, :, [3, 2, 1], :); % from rgb to brg
        batch = permute(batch, [2, 1, 3, 4]);
        batch = single(batch);


        label_tmp = rand( size(batch , 1)/opts.scale , size(batch , 2)/opts.scale , 1 , size(batch , 4) ,'single');
        
        opts.caffe_solver.net.blobs('data').reshape(size(batch)); % reshape blob 'data'
        opts.caffe_solver.net.blobs('labels').reshape(size(label_tmp));
        opts.caffe_solver.net.reshape();
        
        opts.caffe_solver.net.blobs('data').set_data(batch);
        opts.caffe_solver.net.blobs('labels').set_data(label_tmp);
        opts.caffe_solver.net.forward_prefilled();
  
        prob = opts.caffe_solver.net.blobs('proposal_cls_prob').get_data();
        score_fg = prob(:,:,1,:);
        score_fg = squeeze(sum(sum(score_fg ,1) , 2)) /(size(label_tmp,1)*size(label_tmp,2));
        
        score_bg = prob(:,:,2,:);
        score_bg = squeeze(sum(sum(score_bg ,1) , 2)) / (size(label_tmp,1)*size(label_tmp,2));
    
        res_bg = score_fg<score_bg;
        result = result+sum(res_bg);
        
    end
    accurancy_bg = result/n_bg;
        
%%


end

