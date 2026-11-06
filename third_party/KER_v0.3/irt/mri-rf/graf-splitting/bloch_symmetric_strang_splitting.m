function[M]=bloch_symmetric_strang_splitting(u,v,w,d)
%%% Operator splitting based solver for Bloch's equation with relaxation

xdis=d.xdis;
Nx=size(xdis,2);
Nu=size(u,1);
M0=d.M0;
dt=d.dt;
u(u==0)=10^-14;
v(v==0)=10^-14;

%%% Bloch simulation in magnetization domain
gadt = d.gamma*d.dt/2;
B = repmat(gadt*transpose(u-1i*v)*d.B1c, Nx,1);
K = gadt*xdis'*w';
phi = -sqrt(abs(B).^2+K.^2);
D=diag([exp(-1/d.T2*d.relax*dt), exp(-1/d.T2*d.relax*dt), exp(-1/d.T1*d.relax*dt)]);
b=[0;0;d.M0c]-[0;0;d.M0c*exp(-1/d.T1*d.relax*dt)];

M=zeros(3, Nx,Nu+1);
for z=1:Nx
    Mn=M0(:,z);
    M(:,:,1)=M0;
    for n=1:Nu % time loop
        
        phi_j=phi(z,n);
        B1x=real(B(z,n));
        B1y=imag(B(z,n));
        Gx=K(z,n);
        n_j=[B1x; B1y; Gx]./abs(phi_j);
        n_j(isnan(n_j))=0;
        n1=n_j(1);
        n2=n_j(2);
        n3=n_j(3);
        cs=cos(phi_j);
        si=sin(phi_j);
        
        Bd=[n1^2*(1-cs)+cs, n1*n2*(1-cs)-n3*si, n1*n3*(1-cs)+n2*si;
            n2*n1*(1-cs)+n3*si, n2^2*(1-cs)+cs, n2*n3*(1-cs)-n1*si;
            n3*n1*(1-cs)-n2*si, n3*n2*(1-cs)+n1*si, n3^2*(1-cs)+cs];
        

        Mrot=Bd*Mn;
        
        Mt=D*Mrot+b;
        
        Mn=Bd*Mt;
        M(:,z,n+1)=Mn;
    end
end
end

