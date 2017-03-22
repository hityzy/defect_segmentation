function refine_mask = gen_refine_mask(refine_boxes_final , img_result_stage1)
num_of_refine_boxes = size(refine_boxes_final , 1);
refine_mask = zeros(size(img_result_stage1));
for i = 1:num_of_refine_boxes
    rectangle = refine_boxes_final(i,:);
    refine_mask(rectangle(2):(rectangle(2)+rectangle(4)) , rectangle(1):(rectangle(1)+rectangle(3)) , : ) = 1;
end

end