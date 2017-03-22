function [ ims ] = im_crop_regions( img , rectangles)
%%
crop_size = rectangles(1,3)+1;
num_boxes = size(rectangles , 1);
ims = zeros(crop_size, crop_size, 3, num_boxes, 'single');

for i = 1:num_boxes
    rectangle = rectangles(i,:);
    ims(:,:,:,i)  = img(rectangle(2):(rectangle(2)+rectangle(4)) , rectangle(1):(rectangle(1)+rectangle(3)) , : );
end

end
