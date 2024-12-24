function [Anow] = SparseIdentification(Dataset1,Dataset2,h,Xnow,statesize,IdentificationMethod)


if IdentificationMethod==1

%%%%% Sparse Identification    
%%%%% Idetinfication of b1

A1=[Dataset1(1,:)',Dataset1(2,:)'];
B1=(1/h)*(Dataset2(1,:)'-Dataset1(1,:)');
C1=pinv(A1'*A1)*A1'*B1;

%%%%% Idetinfication of b2

A2=[Dataset1(1,:)',Dataset1(2,:)',Dataset1(1,:)'.*Dataset1(3,:)'];
B2=(1/h)*(Dataset2(2,:)'-Dataset1(2,:)');
C2=pinv(A2'*A2)*A2'*B2;

%%%%% Idetinfication of b3

A3=[Dataset1(3,:)',Dataset1(1,:)'.*Dataset1(2,:)'];
B3=(1/h)*(Dataset2(3,:)'-Dataset1(3,:)');
C3=pinv(A3'*A3)*A3'*B3;

%%%%% Calculate Anow
Anow=zeros(statesize,statesize);
Anow(1,1)=1+h*[Xnow(1,1),Xnow(2,1)]*C1;
Anow(2,2)=1+h*[Xnow(1,1),Xnow(2,1),Xnow(1,1)*Xnow(3,1)]*C2;
Anow(3,3)=1+h*[Xnow(3,1),Xnow(1,1)*Xnow(2,1)]*C3;

elseif IdentificationMethod==2

%%%%% Dynamic mode decomposition
%%%%% Idetinfication of b1

A1=[Dataset1(1,:)',Dataset1(2,:)',Dataset1(3,:)'];
B1=(1/h)*(Dataset2(1,:)'-Dataset1(1,:)');
C1=pinv(A1'*A1)*A1'*B1;

%%%%% Idetinfication of b2

A2=[Dataset1(1,:)',Dataset1(2,:)',Dataset1(3,:)'];
B2=(1/h)*(Dataset2(2,:)'-Dataset1(2,:)');
C2=pinv(A2'*A2)*A2'*B2;

%%%%% Idetinfication of b3

A3=[Dataset1(1,:)',Dataset1(2,:)',Dataset1(3,:)'];
B3=(1/h)*(Dataset2(3,:)'-Dataset1(3,:)');
C3=pinv(A3'*A3)*A3'*B3;

%%%%% Calculate Anow
Anow=zeros(statesize,statesize);
Anow(1,1)=1+h*[Xnow(1,1),Xnow(2,1),Xnow(3,1)]*C1;
Anow(2,2)=1+h*[Xnow(1,1),Xnow(2,1),Xnow(3,1)]*C2;
Anow(3,3)=1+h*[Xnow(1,1),Xnow(2,1),Xnow(3,1)]*C3;

elseif IdentificationMethod==3

%%%%% Koopman Approximation 

M=size(Dataset1,2);

% Second order expansion: 1,X1,X2,X3,X1^2,X1X2,X1X3,X2^2,X2X3,X3^2 : 10 basis

%Dataset1_Observe=[ones(1,M);Dataset1;Dataset1(1,:).^2;Dataset1(1,:).*Dataset1(2,:);Dataset1(1,:).*Dataset1(3,:)...
%   ;Dataset1(2,:).^2;Dataset1(2,:).*Dataset1(3,:);Dataset1(3,:).^2];
Dataset1_Observe=[Dataset1];
%Dataset1_Observe=[Dataset1;Dataset1(1,:).*Dataset1(2,:)];

%Dataset2_Observe=[ones(1,M);Dataset2;Dataset2(1,:).^2;Dataset2(1,:).*Dataset2(2,:);Dataset2(1,:).*Dataset2(3,:)...
%    ;Dataset2(2,:).^2;Dataset2(2,:).*Dataset2(3,:);Dataset2(3,:).^2];
Dataset2_Observe=[Dataset2];
%Dataset2_Observe=[Dataset2;Dataset2(1,:).*Dataset2(2,:)];

basis_num=size(Dataset1_Observe,1);
G=zeros(basis_num,basis_num);
A=zeros(basis_num,basis_num);

for i=1:M

G=G+Dataset1_Observe(:,i)*Dataset1_Observe(:,i)';
A=A+Dataset1_Observe(:,i)*Dataset2_Observe(:,i)';

end

G=G/M;
A=A/M;

K=pinv(G)*A;

[Keigenvector,Keigenvalue]=eig(K);

B=zeros(basis_num,statesize);
%B(2,1)=1;
%B(3,2)=1;
%B(4,3)=1;
B(1,1)=1;
B(2,2)=1;
B(3,3)=1;

%Psinow=[1,Xnow(1,1),Xnow(2,1),Xnow(3,1),Xnow(1,1)^2,Xnow(1,1)*Xnow(2,1),Xnow(1,1)*Xnow(3,1),Xnow(2,1)^2,Xnow(2,1)*Xnow(3,1),Xnow(3,1)^2];
Psinow=[Xnow(1,1),Xnow(2,1),Xnow(3,1)];
%Psinow=[Xnow(1,1),Xnow(2,1),Xnow(3,1),Xnow(1,1)*Xnow(2,1)];

Prediction_Now=(inv(Keigenvector)*B)'*Keigenvalue*Keigenvector'*Psinow';

Prediction_Now=real(Prediction_Now);

%%%%% Calculate Anow
Anow=zeros(statesize,statesize);

Anow(1,1)=Prediction_Now(1,1)/Xnow(1,1);
Anow(2,2)=Prediction_Now(2,1)/Xnow(2,1);
Anow(3,3)=Prediction_Now(3,1)/Xnow(3,1);    


end