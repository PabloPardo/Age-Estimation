function [ face ] = detectface( img, margin, detector1, detector2)
    %DETECTFACE: 
    % Given an image, it detects a face using two detectors, 
    % and adding a margin to the face rectagle
    [m, n] = size(img);
    
    bbox = step(detector1, img); 
    face = bbox(find(bbox(:,4)==max(bbox(:,4)),1,'first'),:);
    
    if isempty(face)
        % If no faces are detected, use standard face detector
        bbox = step(detector2, img); 
        face = bbox(find(bbox(:,4)==max(bbox(:,4)),1,'first'),:);
        
        if isempty(face)
            % If still no faces are detected, jump to the next image
            return
        end
    end
    
    % Add margin to the face
    face = [max(face(1)-margin, 0), max(face(2)-margin, 0), ...
            min(face(3)+margin, m), min(face(4)+margin, n)];
    return
end

