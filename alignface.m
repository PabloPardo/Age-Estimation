function [ rot_image, rot_shape ] = alignface( img, eyel, eyer, nose, shape)
%ALIGNFACE Summary of this function goes here
%   Detailed explanation goes here
    p1 = shape(eyel,:);  % Left eye
    p2 = shape(eyer,:);  % Right eye
    p3 = shape(nose,:);  % Center nose
    
    v1 = (p2 - p1)/norm(p2-p1);
    v2 = [1,0];
    
    sig = 1;
    if v1(2) > 0
        sig = -1;
    end
    
    angle = sig*acosd(dot(v1, v2));
    rot = [cosd(angle), sind(angle); -sind(angle), cosd(angle)];
    mid = [size(img,1)/2, size(img,2)/2];
    tr = round(p3*rot) - mid;
    
    rot_shape = shape*rot - repmat(tr, [size(shape,1),1]);
    rot_image = zeros(size(img));
    for i = 1:size(rot_image,1)
        for j = 1:size(rot_image,2)
            new_idx = round(([i,j]-mid)*rot + fliplr(p3));
            if new_idx(1) > 0 && new_idx(2) > 0  && ...
                    new_idx(1) <= size(img,1) && ...
                    new_idx(2) <= size(img,2)
                rot_image(i,j) = img(new_idx(1), new_idx(2));
            end
        end
    end
    rot_image = double(rot_image)/255;
end

