
format short g

THr = 100;
THr1 = 20;

for iii = 1 : 1000
    iii

clearvars -except iii DeepMIMO_dataset iji idx THr THr1

% DeepMIMO_dataset=DeepMIMO_Dataset_Generator;
N = 20;
W = 110; 
L = 121;
Nt = 32 ;  % Number of antennas at BS
Nc = 32 ;  % Number of Subcarriers 
% for iiii=1:25
% iiii


if mod(iii,2)==0
    P=RandomWalk2(N,W,L,DeepMIMO_dataset);
else
    P=RandomWalk1(N,W,L,DeepMIMO_dataset);
end

    



for K=1:N/2
H = DeepMIMO_dataset{1}.user{P(K)}.channel;
L = DeepMIMO_dataset{1}.user{P(K)}.loc;
% Loc = DeepMIMO_dataset{1}.user{P(K)}.loc;
% Channel(iiii,K,:,:) = H;
% Location(iiii,K,:,:,:) = Loc;

 

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
dlmwrite('testframes_O13p5.csv',abs(ADP),'delimiter',',','-append','precision',4);
dlmwrite('testframes_O13p5_Loc.csv',L,'delimiter',',','-append','precision',4);
%dlmwrite('testforpython\testADP_Loc_O13p5.csv',L,'delimiter',',','-append','precision',4);
dlmwrite('testframes_CLADP_O13p5.csv',abs(ADP),'delimiter',',','-append','precision',4);
dlmwrite('testframes_CLADPL_O13p5.csv',abs(ADP),'delimiter',',','-append','precision',4);
dlmwrite('testframes_CLADPNL_O13p5.csv',abs(ADP),'delimiter',',','-append','precision',4);
dlmwrite('testframes_CLADPAddNL_O13p5.csv',abs(ADP),'delimiter',',','-append','precision',4);


% Angle_Delay_Profile((iii-1) * N + K, 1, :, :) = abs(ADP);
% Angle_Delay_Profile_LOS_Blocked(iiii,K,:,:) = ADP;
% Angle_Delay_Profile_ANLOS(iiii,K,:,:) = ADP;

end

for K=N/2+1:N
H = DeepMIMO_dataset{1}.user{P(K)}.channel;
L = DeepMIMO_dataset{1}.user{P(K)}.loc;
% Loc = DeepMIMO_dataset{1}.user{P(K)}.loc;
% Channel(iiii,K,:,:) = H;
% Location(iiii,K,:,:,:) = Loc;

 

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
CLADP = CSI2CLEANADP(H,Nt,Nc,THr);
CLADPL = CSI2CLEANADPLOSBLOCKED(H,Nt,Nc,THr,THr1);
CLADPNL = CSI2CLEANADPNLOSBLOCKED(H,Nt,Nc,THr,THr1);
CLADPANL = ANLOS(H,Nt,Nc,24,24);
% fname = sprintf('test_%d.csv', iji);
dlmwrite('testframes_O13p5.csv',abs(ADP),'delimiter',',','-append','precision',4);
dlmwrite('testframes_O13p5_Loc.csv',L,'delimiter',',','-append','precision',4);
%dlmwrite('testforpython\testADP_Loc_O13p5.csv',L,'delimiter',',','-append','precision',4);
dlmwrite('testframes_CLADP_O13p5.csv',CLADP,'delimiter',',','-append','precision',4);
dlmwrite('testframes_CLADPL_O13p5.csv',CLADPL,'delimiter',',','-append','precision',4);
dlmwrite('testframes_CLADPNL_O13p5.csv',CLADPNL,'delimiter',',','-append','precision',4);
dlmwrite('testframes_CLADPAddNL_O13p5.csv',abs(CLADPANL),'delimiter',',','-append','precision',4);
end
end
