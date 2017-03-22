function [ iou , i , u] = cal_iou(result , Mask)
  intersection = (result)&(Mask);
  union = (result)|(Mask);
  i = sum(sum(intersection));
  u = sum(sum(union));
  iou = i/u;
end

