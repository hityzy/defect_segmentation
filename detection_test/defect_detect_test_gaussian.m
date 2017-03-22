close all;
clear;
clc

class_idx = input('please input class_idx:','s');%'class2';
class = ['class',class_idx];
curt_dir = fileparts(fileparts(mfilename('fullpath')));
cd (curt_dir);

%% only for debug
save_result = 1;
% show_refine_boxes = 1;
% show_refine_image = 1;

%%
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup')); %fileparts = ../
caffe_path = ans;

conf = defect_detect_conf(class , caffe_path);
images = dir(fullfile(pwd , 'dataset' , [class , '_def'] , [class , '_def_img'] , 'test' , ['*.' , conf.ext]));


%% *********************test 1*********************************************************************************
for i = 1:length(images)
%% load images and labels related
mat = load(fullfile(pwd , 'output' , 'images_patch' , class ,'average_image.mat'));
average_image = mat.average_image;
img = single(imread(fullfile(pwd , 'dataset' , [class , '_def'] , [class , '_def_img'] , 'test' , images(i).name)));

%% preprocess check if the size of image is 4*n
if not(mod(size(img , 1),4)==0)||(mod(size(img , 2),4)==0)
         img = modcrop(img, 4);
end
%% prepropose
img = img-average_image;
%% load label mask
imgname = images(i).name();
Mask = imread(fullfile(pwd , 'dataset' , [class , '_def'] , 'Mask' , imgname));
%% change gray image into color image
if size(img,3)==1
    img = cat(3 , img , img , img);
end
%%
% if strcmp(class , 'class7')%guilded filter improve the performance of class7
%         img = imguidedfilter(img,  'NeighborhoodSize' , [17,17] , 'DegreeOfSmoothing' , 110);
% end
%% ***************************stage 1**************************************************
 batch = img;
 batch = batch(:, :, [3, 2, 1], :); % from rgb to brg
 batch = permute(batch, [2, 1, 3, 4]);

net_inputs = {batch};

conf.net1.blobs('data').reshape([size(batch,1) , size(batch , 2) , size(batch , 3) , 1]); % reshape blob 'data'
conf.net1.reshape();

tic;
res = conf.net1.forward(net_inputs);


res = res{1};
res = res(:,:,1);
res = permute(res, [2, 1, 3, 4]);
res = imresize(res , [size(img,1) , size(img , 2)]);

if max(max(res))>=0.8
%    level = graythresh(res);
   level = conf.pos_thresh;
   res_mask=im2bw(res , level);
   res = res.*res_mask; 
   res(res>0) = 1;
else
    res(:,:)=0;
end

res = imfill(res);
% se=strel('disk',10);
% res=imerode(res,se);
% res=imdilate(res,se);

% if strcmp(class , 'class8')%
%     conf.pos_thresh  = 0.95;
%     se=strel('disk',10);
%     res = imclose(res,se);
%     res=imopen(res,se);
% elseif strcmp(class , 'class3')||strcmp(class , 'class4')||strcmp(class , 'class9')
%     se=strel('disk',10);
%     res = imclose(res,se);
%     res=imopen(res,se);
% end
t = toc;
fprintf('time of stage 1  = %f \t' , t)
res_stage1 = res;%save res stage1

%% general result map of stage1
gausFilter = fspecial('gaussian',[5 5],10);
res_stage1=imfilter(res_stage1,gausFilter,'replicate');


%normalize scoremap
max_score = max(max(res_stage1(res_stage1>0)));
min_score = min(min(res_stage1(res_stage1>0)));
res_stage1(res_stage1>0) = (res_stage1(res_stage1>0)-min_score)/(max_score-min_score);

R_channel = zeros(size(res_stage1) , 'single');
R_channel(res_stage1>0) = res_stage1(res_stage1>0);

G_channel = zeros(size(res_stage1) , 'single');
G_channel(res_stage1>0) = (1-res_stage1(res_stage1>0));

heatmap_stage1 = (cat(3 , R_channel , G_channel , zeros(size(res_stage1))));
% heatmap = (cat(3 , zeros(size(R_channel)) , G_channel , zeros(size(res))));

img_result_stage1 = img*0.6+heatmap_stage1*128+img.*not(cat(3,(res_stage1>0),(res_stage1>0),(res_stage1>0)))*0.5;


%% ****************************stage 2*********************************************************************************
%% generate refine boxes
tic
refine_boxes = gen_refine_boxes(res , conf ,  conf.overlap_high);
if isempty(refine_boxes)
    refine_boxes = gen_refine_boxes(res , conf ,  conf.overlap_low);
end

if not(isempty(refine_boxes))
ims = im_crop_regions(img , refine_boxes);
res_2 = [];
nbatchs = ceil(size(ims , 4)/ conf.batch_size); 
for batch_idx = 1:nbatchs
       batch = ims(:,:,:,conf.batch_size*(batch_idx-1)+1:min(conf.batch_size*batch_idx , size(ims , 4)));
       net_inputs = {batch};
% Reshape net's input blobs
       conf.net2.blobs('data').reshape([size(batch,1) , size(batch , 2) , size(batch , 3) , size(batch,4)]); % reshape blob 'data'
       conf.net2.reshape();
       res2 = conf.net2.forward(net_inputs);
       res2 = res2{1};
       [res_w , res_h,~ ,~] = size(res2);
       res2 = sum(sum(res2 , 1) , 2);
       res2 = squeeze(res2(:,:,1 , :))/(res_w*res_h);
       res_2 = [res_2 ; res2];
end
    defect_idx = find(res_2>=conf.pos_thresh2);
    if isempty(defect_idx)
       defect_idx = find(res_2>=conf.pos_thresh*0.9);
    end
fprintf('size of refine boxes1 = %d , size of refine boxes2 = %d \t' , size(refine_boxes , 1) , length(defect_idx));
refine_boxes_final = refine_boxes(defect_idx , :);

if isempty(refine_boxes_final)
    refine_boxes_final = refine_boxes;
end

%% general refine mask and res_stage2
   refine_mask = gen_refine_mask(refine_boxes_final , res);
   res_stage2 = res.*refine_mask;
t = toc;
fprintf('time of stage 2  = %f \n' , t)
else  %refine_boxes is empty
    refine_boxes_final = refine_boxes;
    refine_mask = ones(size(res));
    res_stage2 = res.*refine_mask;
end

%% show result of stage2
% gausFilter = fspecial('gaussian',[5 5],10);
% res_stage2=imfilter(res_stage2,gausFilter,'replicate');

%% ***************************************************************
close all;
gausFilter = fspecial('gaussian',[30 30],1000);
res_stage2_1=imfilter(res_stage2,gausFilter,'replicate');

res_stage2_1 = res_stage2_1.*res_stage2;
imshow(res_stage2_1 , []);
%% **********************************************************************
% gausFilter = fspecial('gaussian',[30 30],1000);
% res_stage4=imfilter(res_stage2,gausFilter,'replicate');
% res_stage4(res_stage4>=0.7) = 1;
% res_stage4(res_stage4<0.7) = 0;
% BW1 = single(bwmorph(res_stage4,'thin',inf));
% SE = strel('disk' , 2);
% BW1 = imdilate(BW1 , SE);
% res_stage2 = res_stage2+BW1/2;
% 
%  gausFilter = fspecial('gaussian',[10 10],10);
%  res_stage3=imfilter(res_stage2,gausFilter,'replicate');
% 
% imshow(res_stage3,[]);
% gausFilter = fspecial('gaussian',[5 5],1000);
% BW2=imfilter(BW1,gausFilter,'replicate');
% BW2 = BW2.*res_stage2;
% imshow(BW2,[]);
% 
% res_stage3 =res_stage3 .*res_stage2;
% imshow(res_stage3,[])
% se=strel('disk',10);
% res_stage2 = imclose(res_stage2,se);

%% ***************************************************************************
%%
%normalize scoremap
max_score = max(max(res_stage2(res_stage2>0)));
min_score = min(min(res_stage2(res_stage2>0)));
res_stage2(res_stage2>0) = (res_stage2(res_stage2>0)-min_score)/(max_score-min_score);

R_channel = zeros(size(res_stage2) , 'single');
R_channel(res_stage2>0) = res_stage2(res_stage2>0);

G_channel = zeros(size(res_stage2) , 'single');
G_channel(res_stage2>0) = (1-res_stage2(res_stage2>0));

heatmap_stage2 = (cat(3 , R_channel , G_channel , zeros(size(res_stage2))));
% heatmap = (cat(3 , zeros(size(R_channel)) , G_channel , zeros(size(res))));

img_result_stage2 = img*0.6+heatmap_stage2*128+img.*not(cat(3,(res_stage2>0),(res_stage2>0),(res_stage2>0)))*0.5;

%% *******************************************************************************************************************
%% save result
if save_result
   result_path = fullfile(pwd , 'results' , 'results_stage1' , class );
   mkdir_if_missing(result_path);
   result_path = fullfile(result_path , images(i).name);
%    gausFilter = fspecial('gaussian',[5 5],10);
%    res=imfilter(res,gausFilter,'replicate');
   seg_result = uint8((ceil(res)*255));
   imwrite(seg_result , fullfile(result_path))
   
   result_path = fullfile(pwd , 'results' , 'results_stage2' , class );
   mkdir_if_missing(result_path);
   result_path = fullfile(result_path , images(i).name);
   res =  res.*refine_mask;
%    gausFilter = fspecial('gaussian',[5 5],10);
%    res=imfilter(res,gausFilter,'replicate');
   seg_result = uint8((ceil(res)*255));
   imwrite(seg_result , fullfile(result_path))
end
%% show init image
img = img+average_image;
subplot(2,3,1);
imshow(uint8(img));

idx = find(images(i).name =='-');
imgname = images(i).name(idx+1:end);
title(imgname);

%% show Mask
subplot(2,3,2);
imshow(uint8(Mask));
title([imgname , ' mask']);
 
%% show result of stage 1
subplot(2,3,3);
imshow(uint8(img_result_stage1+average_image));
title([imgname , ' detection resules of stage1']);

% if show_refine_boxes
% for i =1: size(refine_boxes , 1)  
%      rectangle('Position', refine_boxes(i,:),  'EdgeColor', [1 0 0], 'Linewidth', 0.5); 
% end
% end
%% show refine mask

subplot(2,3,4);
imshow(refine_mask , [])
% imshow(uint8(img_result));
title([imgname , ' refine mask']);
% if show_refine_image
% for i =1: size(refine_boxes_final , 1)  
%      rectangle('Position', refine_boxes_final(i,:),  'EdgeColor', [1 0 0], 'Linewidth', 0.5); 
% end
% end
%% 
%% show show result of stage 2
subplot(2,3,5);
imshow(uint8(img_result_stage2+average_image) , [])
% imshow(uint8(img_result));
title([imgname , 'detection results of tage2']);

%% show final result mask
subplot(2,3,6);
imshow(heatmap_stage2 , [])
title([imgname , 'heatmap']);


% while(1)
%   keydown = waitforbuttonpress;
%   if (keydown == 0)
%     else
%             break;
%   end
% end
end

%% test 2************************************************************************************************
% for i = 1:length(images)
% img = single(imread(fullfile(pwd , 'dataset' , [class , '_def'] , [class , '_def_img'] , 'test' , images(i).name)));
% idx = find(images(i).name =='-');
% imgname = images(i).name(idx+1:end);
% 
% if size(img,3)==1
%     img = cat(3 , img , img , img);
% end
% % 
% if strcmp(class , 'class7')%guilded filter improve the performance of class7
%    img = imguidedfilter(img,  'NeighborhoodSize' , [17,17] , 'DegreeOfSmoothing' , 110);
% end
%  
% 
% x = (1:conf.stride:size(img , 2)-(conf.patch_size-1))'*ones();
% y = (1:conf.stride:size(img , 1)-(conf.patch_size-1))';
% size_x = length(x);
% size_y = length(y);
% x = repmat(x , [1 , size_y])';
% x = x(:);
% y = repmat(y , [size_x , 1]);
% w =(conf.patch_size-1)*ones(size(x)); 
% h = (conf.patch_size-1)*ones(size(y));
% rectangles =[x , y , w , h];
% if strcmp(class , 'class6')
%     idx0 = (rectangles(:,2)<=64) +(rectangles(:,2)>=(size(img , 1)-(conf.patch_size)-10));
%     rectangles(idx0>0 , :)=[];
% end
% 
%  tic;
% ims = im_crop_regions(img , rectangles , class);
% res_ = [];
% nbatchs = ceil(size(ims , 4)/ conf.batch_size); 
% for i = 1:nbatchs
%        batch = ims(:,:,:,conf.batch_size*(i-1)+1:min(conf.batch_size*i , end));
%        net_inputs = {batch};
% % Reshape net's input blobs
% %        conf.net1.reshape_as_input(net_inputs);
%        conf.net.blobs('data').reshape([size(batch,1) , size(batch , 2) , size(batch , 3) , size(batch,4)]); % reshape blob 'data'
%        conf.net.reshape();
%        res = conf.net.forward(net_inputs);
%        res = res{1};
%        [res_w , res_h,~ ,~] = size(res);
%        res = sum(sum(res,1),2);
%        res = squeeze(res(:,:,1 , :))/(res_w*res_h);
%        res_ = [res_ ; res];
% end
%     defect_idx = find(res_>=conf.pos_thresh);
% if isempty(defect_idx)
%     defect_idx = find(res_>=conf.pos_thresh*0.9);
% end
% defect = rectangles(defect_idx , :);
% score = res_(defect_idx);
% time = toc;
% fprintf([imgname ,' cost %fs...\n'] , time);
% 
% 
% %% display results
% score_map = zeros(size(img , 1) , size(img , 2) , 'single');
% for i =1: size(defect , 1)
% score_map(defect(i,2):(defect(i,2)+defect(i,4)) , defect(i,1):(defect(i,1)+defect(i,3))) = ...
%     score_map(defect(i,2):(defect(i,2)+defect(i,4)) , defect(i,1):(defect(i,1)+defect(i,3)))+score(i);
% end
%  Omin = min(score_map(score_map>0));
%  Omax = max(score_map(score_map>0));
%  Nmax = 256;
%  Nmin = 64;
%  if not(Omin==Omax)
%     score_map(score_map>0) = (Nmax-Nmin) / (Omax-Omin) *(score_map(score_map>0)-Omin)+Nmin;
%  else
%     score_map(score_map>0) =Nmax;
%  end
% 
% imshow(uint8(img));
% img_R = img(:,:,1);
% img_R(score_map>0) = img_R(score_map>0)*1+score_map(score_map>0)*0.7;
% img(:,:,1) = img_R;
% waitforbuttonpress;
% imshow(uint8(img));
% text(10,10,imgname,'Color','r', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 10); 
% waitforbuttonpress;
% 
% 
% %% another way to show results
% % imshow(uint8(img));
% % for i =1: size(defect , 1)
% % rectangle('Position', defect(i,:),  'EdgeColor', [0 0.7 0], 'Linewidth', 0.5);
% % end
% % waitforbuttonpress;
% % %
% % 
% % 
% end
%%