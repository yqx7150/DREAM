%%% Bloch simulation with relaxation effects using symmetric operator
%%% splitting
%%% 06.08.2019
%%% Christina Graf c.graf@tugraz.at
clear
clc

%%% init %%%
load('Beispiel1.mat');
d.B1c=57.8704; %f?r Peak B1 von 12.5 muT
d.dt=(8.6409e-7)*2*10^3; %f?r Peak B1 von 12.5 muT und pi/2 in ms
u=u*d.B1c*10^-3;   % in mT
v=v*d.B1c*10^-3;   % in mT
d.B1c=1;
d.G3=18/2; %mT/m, fhwm von 2mm
w=w*d.G3;     % in mT/m
d.G3=1;
d.xdis=linspace(-0.005,0.005,101); % in  m
% u=u*0;
% v=v*0;
w=w*0;

T1=[10^-9 1331 400 832 1420]; % without relax, grey matter, tendons, white matter, muscle
T2=[10^-9 110 5 79.6 31.7];
relax=[0 1 1 1 1]; % 0 without relaxation effects, 1 with relaxation effects
example=["no relax", "grey matter", "tendons", "white matter", "muscle"];

for k=1:5
    d.T1=T1(k);
    d.T2=T2(k);
    d.relax=relax(k);
    %%% non-vectorized simulation %%%
    M_sy=bloch_symmetric_strang_splitting_vectorised(u,v,w,d);
    
    %%% plot %%%
    figure
    plot(d.xdis*1000, M_sy(1,:,end), 'LineWidth', 1.5)
    hold on
    plot(d.xdis*1000, M_sy(2,:,end), 'LineWidth', 1.5)
    plot(d.xdis*1000, M_sy(3,:,end), 'LineWidth', 1.5)
    legend('x','y','z')
    set(gca,'XLim',[-5 5],'XTick',[-5 -2.5 0 2.5 5])
    xlabel('distance [m]', 'FontSize', 20)
    ylabel('magnetization in AU', 'FontSize', 20)
    set(gca, 'YLim', [-1 1], 'YTick', [-1 -0.5 0 0.5 1])
    set(gca, 'FontSize', 20)
    title(example(k));
    disp('press any key to continue')
    pause;   
end
