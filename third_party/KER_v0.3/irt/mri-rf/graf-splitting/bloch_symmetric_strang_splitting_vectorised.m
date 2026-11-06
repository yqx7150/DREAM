function[M]=bloch_symmetric_strang_splitting_vectorised(u,v,w,d)
%%% Operator splitting based solver for Bloch's equation with relaxation

xdis=d.xdis;
% Nx=size(xdis,2);
Nx=d.Nx;
Nu=size(u,1);
M0=d.M0;
dt=d.dt;
% u(u==0)=10^-14;
% v(v==0)=10^-14;

%%% Bloch simulation in magnetization domain
gadt = d.gamma*d.dt/2;
B1 = repmat(gadt*transpose(u-1i*v)*d.B1c, Nx,1);
K = gadt*xdis'*w'*d.G3;
phi = -sqrt(abs(B1).^2+K.^2);

cs=cos(phi);
si=sin(phi);
n1 = real(B1)./abs(phi);
n2 = imag(B1)./abs(phi);
n3 = K./abs(phi);
n1(isnan(n1))=1;
n2(isnan(n2))=0;
n3(isnan(n3))=0;
Bd1 = n1.*n1.*(1-cs)+cs;
Bd2 = n1.*n2.*(1-cs)-n3.*si;
Bd3 = n1.*n3.*(1-cs)+n2.*si;
Bd4 = n2.*n1.*(1-cs)+n3.*si;
Bd5 = n2.*n2.*(1-cs)+cs;
Bd6 = n2.*n3.*(1-cs)-n1.*si;
Bd7 = n3.*n1.*(1-cs)-n2.*si;
Bd8 = n3.*n2.*(1-cs)+n1.*si;
Bd9 = n3.*n3.*(1-cs)+cs;


M=zeros(3, Nx,Nu+1);
M(:,:,1)=M0;
Mt=M0;
D=diag([exp(-1/d.T2*d.relax*dt), exp(-1/d.T2*d.relax*dt), exp(-1/d.T1*d.relax*dt)]);
b=[0;0;d.M0c]-[0;0;d.M0c*exp(-1/d.T1*d.relax*dt)];


for n=1:Nu % time loop
    
    %%% Matrix B out of A_215 -> cartesian coordinates
    %%% [B(1,1) B(1,2) B(1,3) B(2,1) B(2,2) B(2,3) B(3,1) B(3,2) B(3,3)]
    
    Mrot(1,:)=Bd1(:,n)'.*Mt(1,:)+Bd2(:,n)'.*Mt(2,:)+Bd3(:,n)'.*Mt(3,:);
    Mrot(2,:)=Bd4(:,n)'.*Mt(1,:)+Bd5(:,n)'.*Mt(2,:)+Bd6(:,n)'.*Mt(3,:);
    Mrot(3,:)=Bd7(:,n)'.*Mt(1,:)+Bd8(:,n)'.*Mt(2,:)+Bd9(:,n)'.*Mt(3,:);
  
    
    Mt=D*Mrot+b;
    
    Mrot(1,:)=Bd1(:,n)'.*Mt(1,:)+Bd2(:,n)'.*Mt(2,:)+Bd3(:,n)'.*Mt(3,:);
    Mrot(2,:)=Bd4(:,n)'.*Mt(1,:)+Bd5(:,n)'.*Mt(2,:)+Bd6(:,n)'.*Mt(3,:);
    Mrot(3,:)=Bd7(:,n)'.*Mt(1,:)+Bd8(:,n)'.*Mt(2,:)+Bd9(:,n)'.*Mt(3,:);

    
    Mt=Mrot;
    M(:,:,n+1)=Mrot;
end
end

