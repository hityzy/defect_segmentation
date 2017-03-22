function  defect_finetune_hnm( opts , pos_data_train , neg_data_train , pos_data_test , neg_data_test)
%%
res = [] ;
n_pos = size(pos_data_train,4);
n_neg = size(neg_data_train,4);
train_pos_cnt = 0;
train_neg_cnt = 0;

%%
    prev_rng = seed_rand(opts.rng_seed);
    caffe.set_random_seed(opts.rng_seed);
%%

opts.maxiter = opts.maxiter_all*floor(n_pos/opts.batch_pos);

%% extract positive batches
train_pos = [];
remain = opts.batch_pos*opts.maxiter;
while(remain>0)
    if(train_pos_cnt==0)
        train_pos_list = randperm(n_pos)';
    end
    train_pos = cat(1,train_pos,...
        train_pos_list(train_pos_cnt+1:min(end,train_pos_cnt+remain)));
    train_pos_cnt = min(length(train_pos_list),train_pos_cnt+remain);
    train_pos_cnt = mod(train_pos_cnt,length(train_pos_list));
    remain = opts.batch_pos*opts.maxiter-length(train_pos);
end

%% extract negative batches
train_neg = [];
remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter;
while(remain>0)
    if(train_neg_cnt==0)
        train_neg_list = randperm(n_neg)';
    end
    train_neg = cat(1,train_neg,...
        train_neg_list(train_neg_cnt+1:min(end,train_neg_cnt+remain)));
    train_neg_cnt = min(length(train_neg_list),train_neg_cnt+remain);
    train_neg_cnt = mod(train_neg_cnt,length(train_neg_list));
    remain = opts.batchSize_hnm*opts.batchAcc_hnm*opts.maxiter-length(train_neg);
end

%%

% hardnegs = [];
% poss = [];
fprintf('training... \n') ;
for t=1:opts.maxiter
    % ----------------------------------------------------------------------
    % hard negative mining
    % ----------------------------------------------------------------------
    score_hneg = zeros(opts.batchSize_hnm*opts.batchAcc_hnm,1);
    hneg_start = opts.batchSize_hnm*opts.batchAcc_hnm*(t-1);
    for h=1:opts.batchAcc_hnm
        batch = neg_data_train(:,:,:,...
            train_neg(hneg_start+(h-1)*opts.batchSize_hnm+1:hneg_start+h*opts.batchSize_hnm));        
        
        batch = single(batch);
        
        batch = batch(:, :, [3, 2, 1], :); % from rgb to brg
        batch = permute(batch, [2, 1, 3, 4]);
        label_tmp = rand( size(batch , 1)/opts.scale , size(batch , 2)/opts.scale , 1 , size(batch , 4) ,'single');
        
        % Reshape net's input blobs
        opts.caffe_solver.net.blobs('data').reshape(size(batch)); % reshape blob 'data'
        opts.caffe_solver.net.blobs('labels').reshape(size(label_tmp));
        opts.caffe_solver.net.reshape();
        
        opts.caffe_solver.net.blobs('data').set_data(batch);
        opts.caffe_solver.net.blobs('labels').set_data(label_tmp);
        opts.caffe_solver.net.forward_prefilled();
        
        res = opts.caffe_solver.net.blobs('proposal_cls_prob').get_data();
        res = res(:,:,1,:);
        res = squeeze(sum(sum(res ,1) , 2)) / (size(label_tmp,1)*size(label_tmp,2));
        score_hneg((h-1)*opts.batchSize_hnm+1:h*opts.batchSize_hnm) = res;
    end
    [~,ord] = sort(score_hneg,'descend');
    hnegs = train_neg(hneg_start+ord(1:opts.batch_neg));
    im_hneg = neg_data_train(:,:,:,hnegs);

%% one iter SGD update
    batch = cat(4,pos_data_train(:,:,:,train_pos((t-1)*opts.batch_pos+1:t*opts.batch_pos)),...
        im_hneg);
    
    batch = batch(:, :, [3, 2, 1], :); % from rgb to brg
    batch = permute(batch, [2, 1, 3, 4]);
    
%     labels = ones(opts.feature_map_size , opts.feature_map_size , 1, size(batch,4));
    labels = ones( size(batch , 1)/opts.scale , size(batch , 2)/opts.scale , 1 , size(batch , 4) , 'single');
    labels(:,:,:,1:opts.batch_pos) =0; 

    opts.caffe_solver.net.blobs('data').reshape(size(batch)); % reshape blob 'data'
    opts.caffe_solver.net.blobs('labels').reshape(size(labels));
    opts.caffe_solver.net.reshape();

    opts.caffe_solver.net.blobs('data').set_data(batch);
    opts.caffe_solver.net.blobs('labels').set_data(labels);
    opts.caffe_solver.step(1);
    

    iter = opts.caffe_solver.iter();
    %% show result per 20 iter
    if (mod(iter , 1)==0)
     loss = opts.caffe_solver.net.blobs('loss_cls').get_data();
     accurancy = opts.caffe_solver.net.blobs('accuarcy').get_data();
     fprintf('training batch %3d of %3d ...loss = %f  and accurancy = %f\n', iter, opts.maxiter , loss , accurancy) ;
     fprintf(opts.fid_loss , '%f\n' , loss);
     fprintf(opts.fid_accurancy_train , '%f\n' , accurancy);
     
    end
   %% save models per 800iter
   if (mod( iter , 500)==0)
       file_name = sprintf('iter_%d', iter);
       model_path = fullfile(opts.model_path , file_name );
       opts.caffe_solver.net.save(model_path);
       fprintf('caffemodel is saved as %s\n', model_path);
     if(opts.do_val)
         fprintf('iter = %d  doing validation...   ',iter);
         [ accurancy_fg , accurancy_bg ]=do_validation(opts , pos_data_test , neg_data_test);
         fprintf('accurancy of fg is %f  ;  accurancy of bg is %f\n',  accurancy_fg , accurancy_bg);
         fprintf(opts.fid_accurancy_vali , '%f , %f\n',  accurancy_fg , accurancy_bg);
     end
   end
    
    
end % next batch

%%
file_name = sprintf('iter_%d_final', iter);
model_path = fullfile(opts.model_path , file_name );
opts.caffe_solver.net.save(model_path);
fprintf('caffemodel is saved as %s\n', model_path);
caffe.reset_all();
rng(prev_rng);
end

