clear;
clc;

%%% parameters for Lorenz system
sigma=10;
beta=8/3;
rou=28;

f1=@(x1,x2,x3) sigma*(x2-x1);
f2=@(x1,x2,x3) x1*(rou-x3)-x2;
f3=@(x1,x2,x3) x1*x2-beta*x3;

initialpoint=[1,1,1]';
goalpoint=[-sqrt(72),-sqrt(72),27]';

%%% prediction parameters
h=0.01;
statesize=3;
inputsize=1;
Tend=150;
NSparse=100;

X_record=[];
X_record(:,end+1)=initialpoint;
U_record=[];

for i=1:NSparse

  Xnow=X_record(:,end);  
  X_record(:,end+1)=Xnow+h*[f1(Xnow(1,1),Xnow(2,1),Xnow(3,1));f2(Xnow(1,1),Xnow(2,1),Xnow(3,1));f3(Xnow(1,1),Xnow(2,1),Xnow(3,1))];
  U_record(:,end+1)=zeros(inputsize,1);

end

%%% recursive optimization
Xnow=X_record(:,end);
R=goalpoint;
N=5; % predicted number of steps

%%% optimization weight coefficient
S=2*eye(statesize);
Q=eye(statesize);
%% T=0.1*eye(inputsize);
T=50*eye(inputsize);

Q_=[];
for i=1:N+1

    if i~=N+1
    Q_=blkdiag(Q_,Q);
    else
    Q_=blkdiag(Q_,S);
    end

end

N_=[];
for i=1:N+1

    if i~=N+1
    N_=[N_,Q];
    else
    N_=[N_,S];
    end

end

N_=-2*(R')*N_;

T_=[];
for i=1:N

    T_=blkdiag(T_,T);
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U_limit=h*[-5*ones(N,1),5*ones(N,1)];

Dataset1=X_record(:,1:NSparse);    %%% Preimage data set for Sparse identification
Dataset2=X_record(:,2:NSparse+1);  %%% Image data set for Sparse identification

IdentificationMethod=2;  %%% 1:Sparse Identification; 2: Dynamic Mode Decomposition; 3: Koopman Approximation
ObservationNoiseIntensity=0.5;  %%% Noise Intensity=1 for the comparasion 

for i=1:(Tend/h)

  i  

  [Anow] = SparseIdentification(Dataset1,Dataset2,h,Xnow,statesize,IdentificationMethod);
  
  Bnow=zeros(statesize,inputsize);
  Bnow(1,1)=1;

  [J1,J2] = MPCmatrix(statesize,inputsize,N,Anow,Bnow,Q_,N_,T_,Xnow);
  
  %%% Quadratic programming optimization
  % x = quadprog(H,f,A,b,Aeq,beq,lb,ub)

  H=(1/2)*J1;
  f=(J2)';
  A=[];
  b=[];
  Aeq=[];
  beq=[];
  lb=U_limit(:,1);
  ub=U_limit(:,2);

  [U_local] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
  U_now=U_local(1,1);

  DataPreimageNow=Xnow;  %%%  

  Xnow=Xnow+h*[f1(Xnow(1,1),Xnow(2,1),Xnow(3,1));f2(Xnow(1,1),Xnow(2,1),Xnow(3,1));f3(Xnow(1,1),Xnow(2,1),Xnow(3,1))];
  
  Xnow=Xnow+Bnow*U_now;

  DataImageNow=Xnow-Bnow*U_now;  %%%

  X_record(:,i+1)=Xnow;

  U_record(:,i)=U_local(1,1);


  if sqrt(R-Xnow)>0.001

    Bh1=ObservationNoiseIntensity*sqrt(h)*randn(statesize,1);
    Bh2=ObservationNoiseIntensity*sqrt(h)*randn(statesize,1);

    Dataset1=Dataset1(:,2:end);
    Dataset1(:,end+1)=DataPreimageNow+Bh1;
    Dataset2=Dataset2(:,2:end);
    Dataset2(:,end+1)=DataImageNow+Bh2;

  end


end

tlineX=h*(1:1:size(X_record,2));
tlineU=h*(1:1:size(U_record,2));

subplot(3,1,1)
set(gcf,'color','white');
plot(tlineX,X_record(1,:),'g');
hold on;
plot(tlineX,X_record(2,:),'b');
hold on;
plot(tlineX,X_record(3,:),'r');

xlabel('t');
xlim([0,Tend]);
legend('X_1','X_2','X_3');

subplot(3,1,2)
plot(tlineU,U_record);
xlabel('t');
ylabel('U');

subplot(3,1,3)
plot3(X_record(1,:),X_record(2,:),X_record(3,:));
hold on;
scatter3(initialpoint(1,1),initialpoint(2,1),initialpoint(3,1),'g','filled');
hold on;
scatter3(goalpoint(1,1),goalpoint(2,1),goalpoint(3,1),'r','filled');

xlabel('x_1');
ylabel('x_2');
zlabel('x_3');
legend('Controlled trajectory','Start point','Target point');
