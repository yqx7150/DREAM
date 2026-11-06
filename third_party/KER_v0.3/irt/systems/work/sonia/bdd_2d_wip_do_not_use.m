%% 2D Branchless Distance Driven 
% By Sonia Minseo Kim 
% Aug 2022

clear; clc;

%% Geometry Definitions

% Geometry
geo.DSD = 100000;       % Distance from source to detector (mm)
geo.DS0 =  50000;       % Distance from source to iso-center
geo.pSize = 1;          % Square pixel size (mm)
geo.dSize = 0.5;        % Square detector size (mm)

geo.nPix = 254;         % Number of pixels elements (Row and Col)
geo.nDet = 2000;        % Number of detector elements

% Angle
deg = 1;
geo.theta = deg2rad(0:deg:360-deg);

% Make the Phantom Image
phantomImg = phantom(geo.nPix ,geo.nPix);
phantomImg = mat2gray(phantomImg); % set min of image to zero

% Projection
sinogramB = projection(phantomImg,geo);

%% Projection

function sinogram = projection(phantom, geo)

DSD = geo.DSD;      
DS0 = geo.DS0;      
pSize = geo.pSize;    
dSize = geo.dSize;    
nPix = geo.nPix;       
nDet = geo.nDet;     
theta = geo.theta; 

% Detector Boundaries
detX = (-(nDet/2):(nDet/2)) .* dSize;
detY = (-(DSD-DS0)-(dSize/2)) .* ones(1,nDet+1);

% Pixel Boundaries 
[pixelX,pixelY] = meshgrid(-(nPix/2):(nPix/2));
pixelX = pixelX .* pSize;
pixelY = flipud(pixelY - pSize/2);

sinogram = zeros(size(theta,2), nDet);

% For each projection image
for proj = 1:size(theta,2)
    
    %{ 
        calculating system matrix coefficient
        1. Define the beta cases 
        2. Determine which axis to project boundaries
        3. overlap kernel (IID steps)
    %}
    
    beta = theta(proj);

    % Tube rotation
    rtubeX = -DS0*sin(beta);
    rtubeY = DS0*cos(beta);

    % Detector rotation
    rdetX = detX.*cos(beta) - detY.*sin(beta);
    rdetY = detX.*sin(beta) + detY.*cos(beta);

    if ((beta >= pi/4) && (beta <= 3*pi/4) || (beta >= 5*pi/4) && (beta <= 7*pi/4))
        axisCase = 1; % map on x axis
    else
        axisCase = 0; % map on y axis
    end

    if (axisCase)
        detm = map2x(rtubeX,rtubeY,rdetX,rdetY);
        pixm = map2x(rtubeX,rtubeY,pixelX,pixelY);
        img = phantom;
    else 
        detm = map2y(rtubeX,rtubeY,rdetX,rdetY);
        pixm = fliplr(map2y(rtubeX,rtubeY,pixelX,pixelY)');
        img = fliplr(phantom');
    end

    for n=1:nDet
         det_mid = (detm(n) + detm(n+1))/2;
    end

    if(axisCase)
        % pixel intersection calculation
        cosine = rtubeY./sqrt((rtubeX-det_mid).^2 + (rtubeY-0).^2);
        L = abs(pSize/cosine);
          for n=1:nDet
              L1(n)=sqrt((rtubeX-det_mid).^2+(rtubeY-0).^2)...
              /sqrt((rtubeX-detm(n)).^2+(rtubeY-0).^2);
          end
    else
        sine = (rtubeX-det_mid)./sqrt((rtubeX-det_mid).^2 + (rtubeY-0).^2);
        L = abs(pSize/sine);
          for n=1:nDet
              L1(n)=sqrt((rtubeX-0).^2+(rtubeY-det_mid).^2)...
              /sqrt((rtubeX-0).^2+(rtubeY-detm(n)).^2); % denominator = hypotenuse
          end           
    end     
    L = L ./ L1;

    sinoTmp = zeros(1,nDet);

    for row = 1:nPix

        rowm = pixm(row,:); % mapped row from image
        
        inc = 1;
        currentPix = rowm(row);
        currentDet = detm(row);
        detVal = currentDet; % what is detVal

        while (currentDet <= detm(end) && row+inc <= nPix)
            pixBound = rowm(row+inc);
            detBound = detm(row+inc);

            if (detBound <= pixBound)
                detVal = detVal + (detBound - currentDet)*currentPix;
                currentDet = detBound;
                inc = inc + 1;
            else
                detVal = detVal + (pixBound - currentPix)*currentPix;
                currentPix = pixBound;
                inc = inc + 1;
            end
        end

        % Digital integration of source signal (projection: pixel values)
        pixelSize = (pixm(1,end)-pixm(1,1))/nPix;
        detSize = (detm(1,end)-detm(1,1))/nDet;

        P_pj = integrateDig(img(row,:), pixelSize);

        % Linear interpolation from pixel to detector position
        P_dk = interp1(rowm,P_pj,detm,'linear',0);

        % Digital differentiation of the interpolated signal values
        while(detBound<rowm(end))
            sinoTmp(currentDet) = sinoTmp(currentDet) + (P_dk(currentDet)-P_dk(row+inc))/detSize;
            currentDet = row+inc;
        end
        sinoTmp(currentDet) = sinoTmp(currentDet) + (P_pj(end)-P_dk(currentDet))/detSize; 

    end

    sinogram(proj,:) = sinoTmp .* L;

end


end

%% Digital Integration function
function [P_pj] = integrateDig(p_v,pixelSize)

n_pixel = size(p_v,2);

P_x = 0;
P_pj = zeros(1,n_pixel+1);

for pj=1:n_pixel
    
   P_x = P_x + p_v(pj) * pixelSize;
   
   P_pj(pj+1) = P_x;
   
end

P_pj(1) = -0.5 * P_x;
P_pj = P_pj-P_pj(1);

end

%% Map Y
% Function that map detector or pixel bounderies onto Y axis
function [y] = map2y(x1,y1,x2,y2)
    y=y1-x1.*(y1-y2)./(x1-x2);
end

%% Map X
% Function that map detector or pixel bounderies onto X axis
function [x] = map2x(x1,y1,x2,y2)
    x=x1-y1.*(x1-x2)./(y1-y2);
end