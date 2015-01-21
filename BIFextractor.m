function [ L2 ] = BIFextractor( img, param )
%BIFEXTRACTOR 
    Pscale = param.Pscale;
    Pwavelength = param.Pwavelength;
    Pfiltersize = param.Pfiltersize;
    shortestside = param.shortestside;
    NumbofOrient = param.NumbofOrient;
    Numberofscales = param.Numberofscales;
    maxneigh = param.maxneigh; 
    Ns = param.Ns;
    L2stepsize = param.L2stepsize;        
        
    Gabor=create_gabors(Numberofscales,NumbofOrient,Pscale,Pwavelength,Pfiltersize);
%     displaygabors(Gabor)

    %%
    [m, n] = size(img);
    if (m<n)
        img = imresize(img,[shortestside NaN]);                 % resize image.
    else
        img = imresize(img,[NaN shortestside]);
    end

    % L1 LAYER  (NORMALIZED DOT PRODUCT OF GABORS FILTERS ON LOCAL PATCHES OF IMAGE "A" AT EVERY POSSIBLE LOCATIONS AND SCALES)
    L1 = L1_layer(Gabor, img);

    % L2 LAYER: LOCAL MAX POOLING OF L1 LAYER OVER LOCAL POSITION AT ALL SCALES AND ALL ORIENTATIONS
    % THE MAXIMUM POOLING SLIDING WINDOW SIZES ARE CONTAINED IN "maxneigh" AND "L2stepsize" INDICATES THE CORRESPONDING STEPSIZE
    L2 = L2_layer(L1, L2stepsize, maxneigh, Ns);
    
    L2 = L2(:);
end

