%% 2D Branchless Distance Driven 
% Sonia Minseo Kim 
% 2022

clear; clc;

%% Geometry Definitions

% Geometry
geo.DSD = 949;       % Distance from source to detector (mm)
geo.DS0 =  541;       % Distance from source to iso-center
geo.pSize = 1;          % Square pixel size (mm)
geo.dSize = 1.0239;        % Square detector size (mm)

geo.nPix = 512;         % Number of pixels elements (Row and Col)
geo.nDet = 888;        % Number of detector elements

% Angle
deg = 1;
geo.angle = deg2rad(0:deg:360-deg);

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
angle = geo.angle; 

% Detector Boundaries
detX = (-(nDet/2):(nDet/2)) .* dSize;
detY = (-(DSD-DS0)-(dSize/2)) .* ones(1,nDet+1);

% Pixel Boundaries 
[pixelX,pixelY] = meshgrid(-(nPix/2):(nPix/2));
pixelX = pixelX .* pSize;
pixelY = flipud(pixelY - pSize/2);

sinogram = zeros(size(angle,2), nDet);

% For each projection image
for proj = 1:size(angle,2)
    
    %{ 
        calculating system matrix coefficient A
        1. Define the beta cases 
        2. Determine which axis to project boundaries
        3. overlap kernel (branching or IID steps)
    %}
    
    beta = angle(proj); % angle from beam to y axis

    % Tube rotation
    rtubeX = DS0*sin(beta);
    rtubeY = DS0*cos(beta);

    % Detector rotation
    rdetX = detX.*cos(beta) - detY.*sin(beta);
    rdetY = detX.*sin(beta) + detY.*cos(beta);

    if ((beta >= pi/4) && (beta <= 3*pi/4) || (beta >= 5*pi/4) && (beta <= 7*pi/4))
        axisCase = 0; % map on y axis
    else
        axisCase = 1; % map on x axis
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

    % scaling factor calculation
    detSize = diff(detm);
    if(axisCase) %if map on x-axis
          for n=1:nDet 
              det_mid = (detm(n) + detm(n+1))/2;
              theta = atan((rtubeX+det_mid)/rtubeY); %theta is the angle of the ray to det_mid
              L(n) = abs(pSize./(detSize(n).*cos(theta)));
          end
    else %if map on y-axis
          for n=1:nDet
              det_mid = (detm(n) + detm(n+1))/2;
              theta = atan((rtubeY+det_mid)/rtubeX);
              L(n) = abs(pSize./(detSize(n).*sin(theta)));
          end           
    end     
    
    sinoTmp = zeros(1,nDet);

    for row = 1:nPix

        rowm = pixm(row,:); % mapped row from image
        
        pixelSize = (pixm(1,end)-pixm(1,1))/nPix;
        detSize = diff(detm);

        detInd = 1;
        pixInd = 1;
        
        % Find first detector index overlap with mapped pixel
        if(rowm(1) > detm(end))
            continue; % warning 
        end

        if ((detm(detInd)-rowm(pixInd)< -detSize(detInd)))  
            while((detm(detInd)-rowm(pixInd)< -detSize(detInd)) && detInd <= nDet && pixInd <= nPix)            
                detInd = detInd + 1;            
            end
        
        elseif (detm(detInd)-rowm(pixInd)> pixelSize)
            while(detm(detInd)-rowm(pixInd)> pixelSize && detInd <= nDet && pixInd <= nPix)            
                pixInd = pixInd + 1;            
            end
        end
        
        detInc = 1; %to locate the next index
        pixInc = 1;
        currentPix = rowm(row); %first pixel in the row
        currentDet = detm(detInd); %first detector in the view

        % overlap calculation
        % procedure resembles convolution?
        while (currentDet <= detm(end) && row+pixInc <= nPix && detInd+detInc <= nDet)
            pixBound = rowm(row+pixInc);
            detBound = detm(detInd+detInc); %detector boundary (i.e. next detector position)
            
            %Case 1
            if (detBound <= pixBound && currentDet >= currentPix)
                sinoTmp(detInd) = sinoTmp(detInd) + (detBound - currentDet)*currentDet;
                currentDet = detBound;
                currentPix = pixBound;

            %Case 2
            elseif (detBound > pixBound && currentDet >= currentPix && currentDet < pixBound)
                sinoTmp(detInd) = sinoTmp(detInd) + (pixBound - currentDet)*currentDet;
                currentDet = detBound;
                currentPix = pixBound;

            %Case 3
            elseif (currentDet < currentPix && detBound > currentPix && detBound < pixBound)
                sinoTmp(detInd) = sinoTmp(detInd) + (detBound - currentPix)*currentDet;
                currentDet = detBound;
                currentPix = pixBound;
                
            %Case 4
            elseif (currentDet < currentPix && detBound > pixBound)
                sinoTmp(detInd) = sinoTmp(detInd) + (pixBound - currentPix)*currentDet;
                currentDet = detBound;
                currentPix = pixBound;
            
            end

                detInc = detInc + 1;
                pixInc = pixInc + 1;
        end
        

        %{
        % detm(end) --> last detector in the view
        % Find first detector overlap mapped with pixel mapped (Case 1)
        if(detm(detInd)-rowm(pixInd)<=detSize)
            while((detm(detInd)-rowm(pixInd)<=detSize))            
                detInd = detInd + 1;            
            end
        else
        % Find first pixel overlap mapped with detector mapped (Case 2)           
            if(detm(detInd)-rowm(pixInd)>pixelSize)
                while(detm(detInd)-rowm(pixInd)>pixelSize)            
                    pixInd = pixInd + 1;            
                end
            end
        end

        %}
        
        %{
        IID steps
        % Digital integration of source signal (projection: pixel values)
        P_pj = integrateDig(img(row,:), pixelSize);

        % Linear interpolation from pixel to detector position
        P_dk = interp1(rowm,P_pj,detm,'linear',0);

        % Digital differentiation of the interpolated signal values
          % Update new detector value starting from detInd
        while(detInd<nDet && detInd>0)
            sinoTmp(detInd) = sinoTmp(detInd) + (P_dk(detInd+1)-P_dk(detInd))/detSize(detInd);
            detInd = detInd+1;
        end
        sinoTmp(detInd) = sinoTmp(detInd) + (P_pj(end)-P_dk(detInd))/detSize(detInd);
        %}

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
% Function that map detector or pixel boundaries onto Y axis
function [y] = map2y(x1,y1,x2,y2)
    y=y1-x1.*(y1-y2)./(x1-x2);
end

%% Map X
% Function that map detector or pixel boundaries onto X axis
function [x] = map2x(x1,y1,x2,y2)
    x=x1-y1.*(x1-x2)./(y1-y2);
end

%% Function to map new detector values
% May implement it later if necessary
%{
function [newDetVal] = forward(pixm,nPix,detm,nDet)
    pixelSize = (pixm(1,end)-pixm(1,1))/nPix;
    detSize = (detm(1,end)-detm(1,1))/nDet; %det vals are unevenly spaced?

    detInd = 1;
    % Find first detector index overlap with mapped pixel
    while(detm(detInd)-rowm(1)<=detSize)            
       detInd = detInd + 1;            
    end

    % Update new detector value starting from detInd
    while (detInd <= nDet)
        newDetVal(detInd) = detm(detInd)+(Pdk(detInd+1)-Pdk(detInd))/detSize;
        sinoTmp(detInd) = sinoTmp(detInd)+(Pdk(detInd+c2)-Pdk(detInd))/deltaDetm;
        detInd = detInd + detIinc;
    end
    

end
%}



