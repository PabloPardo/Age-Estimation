function [ face, points ] = FGNETface(img, path)
points_path = strrep(path, 'all_img', 'points');
points_path = strrep(points_path, 'JPG', 'pts');
points_path = strrep(points_path, 'Gf', '');

points = dlmread(points_path, ' ', [3, 0, 70, 1]);

m_x = min(points(:,1));
m_y = min(points(:,2));
M_x = max(points(:,1));
M_y = max(points(:,2));
dx = M_x - m_x;
dy = M_y - m_y;
max_d = max(dx, dy);

crop = [m_x, m_y, max_d, max_d];

if size(img,3) > 1
    img = rgb2gray(img);
end

% Crop and resize the image to the face
face = imresize(imcrop(img, crop), [200, 200]);

% Rescale the points
points(:,1) = (points(:,1) - m_x)*200/max_d;
points(:,2) = (points(:,2) - m_y)*200/max_d;

end