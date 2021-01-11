% Added NLOS  

function ADP=ANLOS(H,Nt,Nc,xbios,ybios)
% H=DeepMIMO_dataset{1}.user{(4000)}.channel;
% Nt = 64 ;  % Number of antennas at BS
% Nc = 32 ;  % Number of Subcarriers 

%% Constructing V
V = zeros(Nt,Nt);

for i = 1 : Nt
    for j = 1 : Nt
        V(i,j) = exp(-1i * 2 * pi * i * ( j - Nt/2) / Nt );
    end 
end 

%% Constructing F
F = zeros(Nc,Nc);

for i = 1 : Nc
    for j = 1 : Nc
        F(i,j) = exp(1i * 2 * pi * i * j / Nc );
    end 
end 


%% Constructing Angle-delay profiles 

ADP = V' * H * F ;

%% finding LOS path

% figure(1)
% imagesc(abs(ADP))

[a,b]=find(abs(ADP) == max(max(abs(ADP))));
maxADP = max(max(abs(ADP)));

% xbios=randi([floor(1*Nt/4) floor(3*Nt/4)],1,1);
% ybios=randi([floor(1*Nt/4) floor(3*Nt/4)],1,1);

ADP(mod(a+xbios,Nt)+(mod(a+xbios,Nt)==0)*Nt,mod((b+ybios-5:1:b+ybios+5),Nc)+(mod(b+ybios-5:1:b+ybios+5,Nc)==0)*Nc) = maxADP/10;
ADP(mod(a+xbios+(-5:1:5),Nt)+(mod(a+xbios+(-5:1:5),Nt)==0)*Nt,mod(b+ybios,Nc)+(mod(b+ybios,Nc)==0)*Nc) = maxADP/10;
ADP(mod(a+xbios,Nt)+(mod(a+xbios,Nt)==0)*Nt,mod(b+ybios,Nc)+(mod(b+ybios,Nc)==0)*Nc) = maxADP/2;

% ADP(mod(a+1,Nt)+(mod(a+1,Nt)==0)*Nt,mod(b+1,Nc)+(mod(b+1,Nc)==0)*Nc)=0;
% ADP(mod(a+1,Nt)+(mod(a+1,Nt)==0)*Nt,mod(b-1,Nc)+(mod(b-1,Nc)==0)*Nc)=0;
% ADP(mod(a-1,Nt)+(mod(a-1,Nt)==0)*Nt,mod(b-1,Nc)+(mod(b-1,Nc)==0)*Nc)=0;
% ADP(mod(a-1,Nt)+(mod(a-1,Nt)==0)*Nt,mod(b+1,Nc)+(mod(b+1,Nc)==0)*Nc)=0;

% figure(2)
% imagesc(abs(ADP))

end