function L2 = L2_layer(L1,L2stepsize,maxneigh, Ns)
% L2 LAYER: Local MAX of L1 layer over local position at all scales and all orientation
% and local STD of maximum map.

% INPUTS: 
% Layer L1

% OUTPUT:
% Layer L2: reduced version of L1 by selecting maxima in spatial
% neighborhoods of size maxneigh in step size "L2stepsize" and over 2 adjacent scales.


[a, b, c, d] = size(L1);
Numberofscales = c;
NumbofOrient = d;
NumGrid_i = a/Ns;
NumGrid_j = b/Ns;

L2 = zeros(2*Ns-1, Ns, c-1, d);    
std = zeros(2*Ns-1, Ns);

%% MAX + STD
for s=1:Numberofscales-1                         % loop on scales  
    for o=1:NumbofOrient                         % loop on orientation
        
        % MAX Pooling
        m1 = minmaxfilt(L1(:,:,s,o), [maxneigh(s), maxneigh(s)],'max','same');
        m2 = minmaxfilt(L1(:,:,s+1,o), [maxneigh(s), maxneigh(s)],'max','same');  
        F = max(m1,m2);
        
        % STD Pooling
        ii = 1;
        jj = 1;
        for g=1:(2*Ns-1)*Ns                      % loop on the grid regions
            idx_i = floor((ii-1)*NumGrid_i/2 + 1: min(ii*NumGrid_i/2 + NumGrid_i/2, a));
            idx_j = floor((jj-1)*NumGrid_j + 1: min(jj*NumGrid_j, b));
            
            std(g) = sqrt(1/(Ns*Ns) * sum(sum((F(idx_i,idx_j) - mean2(F(idx_i,idx_j))).^2)));
            
            if mod(ii, 2*Ns-1) == 0
                ii = 1;
                jj = jj + 1;
            else
                ii = ii + 1;
            end
        end
        L2(:,:,s,o) = std;
    end
end
L2 = L2(1:L2stepsize:end,1:L2stepsize:end,:,:);



end