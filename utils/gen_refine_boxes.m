function [ refine_boxes ] = gen_refine_boxes( res , conf , overlap)

area_patch = conf.patch_size*conf.patch_size;
overlap_thres = overlap*area_patch;

x = (1:conf.stride:size(res , 2)-(conf.patch_size-1))'*ones();
y = (1:conf.stride:size(res , 1)-(conf.patch_size-1))';
size_x = length(x);
size_y = length(y);
x = repmat(x , [1 , size_y])';
x = x(:);
y = repmat(y , [size_x , 1]);
w =(conf.patch_size-1)*ones(size(x)); 
h = (conf.patch_size-1)*ones(size(y));
rectangles =([x , y , w , h]);

num_boxes = size(rectangles , 1);
refine_idx = zeros(num_boxes,1,'single');

for i = 1:num_boxes
    sub_rectangle = rectangles(i,:);
    sub_res  = res(sub_rectangle(2):(sub_rectangle(2)+sub_rectangle(4)) , sub_rectangle(1):(sub_rectangle(1)+sub_rectangle(3)) , : );
    if sum(sum(sub_res))>overlap_thres
        refine_idx(i)=1;
    end
end

refine_boxes = rectangles(refine_idx>0 , :);

%% for test
% imshow(cat(3,res,res,res),[]);
%  for i =1: size(refine_boxes , 1)  
%      rectangle('Position', refine_boxes(i,:),  'EdgeColor', [1 0 0], 'Linewidth', 0.5); 
%  end
% 
% end

