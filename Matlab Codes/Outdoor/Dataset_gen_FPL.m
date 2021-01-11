
format short g

THr = 100;
THr1 = 20;

for iii = 1 : 1000
    iii

clearvars -except iii DeepMIMO_dataset iji idx THr THr1

% DeepMIMO_dataset=DeepMIMO_Dataset_Generator;
N = 20;
W = 110; 
L = 181;
Nt = 64 ;  % Number of antennas at BS
Nc = 64 ;  % Number of Subcarriers 
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
%CLADP = CSI2CLEANADP(H,Nt,Nc,THr);
%CLADPL = CSI2CLEANADPLOSBLOCKED(H,Nt,Nc,THr,THr1);
%CLADPNL = CSI2CLEANADPNLOSBLOCKED(H,Nt,Nc,THr,THr1);
% fname = sprintf('test_%d.csv', iji);
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



% Angle_Delay_Profile((iii-1) * N + K, 1, :, :) = abs(ADP);
% Angle_Delay_Profile_LOS_Blocked(iiii,K,:,:) = ADP;
% Angle_Delay_Profile_ANLOS(iiii,K,:,:) = ADP;

end
end

%{
for iii = 1001 : 2000
    iii
%     pause(5)
clearvars -except iii DeepMIMO_dataset iji idx THr THr1

% DeepMIMO_dataset=DeepMIMO_Dataset_Generator;
N = 20;
W = 1100; 
L = 181;
Nt = 64 ;  % Number of antennas at BS
Nc = 64 ;  % Number of Subcarriers 
% for iiii=1:25
% iiii
P=RandomWalk2(N,W,L,DeepMIMO_dataset);


for K=1:N
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
%CLADP = CSI2CLEANADP(H,Nt,Nc,THr);
%CLADPL = CSI2CLEANADPLOSBLOCKED(H,Nt,Nc,THr,THr1);
%CLADPNL = CSI2CLEANADPNLOSBLOCKED(H,Nt,Nc,THr,THr1);
% fname = sprintf('test_%d.csv', iji);
dlmwrite('paperO1\testADP_O13p5.csv',abs(ADP),'delimiter',',','-append','precision',4);
dlmwrite('paperO1\testADP_O13p5_Loc.csv',L,'delimiter',',','-append','precision',4);
%dlmwrite('testforpython\testADP_Loc_O13p5.csv',L,'delimiter',',','-append','precision',4);
%dlmwrite('testforpython\testADP_CLADP_O13p5.csv',CLADP,'delimiter',',','-append','precision',4);
%dlmwrite('testforpython\testADP_CLADPL_O13p5.csv',CLADPL,'delimiter',',','-append','precision',4);
%dlmwrite('testforpython\testADP_CLADPNL_O13p5.csv',CLADPNL,'delimiter',',','-append','precision',4);



% Angle_Delay_Profile((iii-1) * N + K, 1, :, :) = abs(ADP);
% Angle_Delay_Profile_LOS_Blocked(iiii,K,:,:) = ADP;
% Angle_Delay_Profile_ANLOS(iiii,K,:,:) = ADP;

end
end
% end
% end

% N_distortion = 30;
% S_distortion = randi([1,300-N_distortion],1,1);
% xbios=randi([floor(1*Nt/4) floor(3*Nt/4)],1,1);
% ybios=randi([floor(1*Nt/4) floor(3*Nt/4)],1,1);

% for K = S_distortion : S_distortion + N_distortion
%     Angle_Delay_Profile_LOS_Blocked(iiii,K,:,:) = LOS_Blockage(DeepMIMO_dataset{1}.user{P(K)}.channel,Nt,Nc);
%     Angle_Delay_Profile_ANLOS(iiii,K,:,:) = ANLOS(DeepMIMO_dataset{1}.user{P(K)}.channel,Nt,Nc,xbios,ybios);
% end
% end

% for iji=6:10
% for iiii=5001:10000
%     iiii
% P=RandomWalk2(N,W,L,DeepMIMO_dataset);
% 
% for K=1:N
% 
% H = DeepMIMO_dataset{1}.user{P(K)}.channel;
% % Loc = DeepMIMO_dataset{1}.user{P(K)}.loc;
% % Channel(iiii,K,:,:) = H;
% % Location(iiii,K,:,:,:) = Loc;
% 
%  
% 
% %% Constructing V
% V = zeros(Nt,Nt);
% 
% for i = 1 : Nt
%     for j = 1 : Nt
%         V(i,j) = exp(-1i * 2 * pi * i * ( j - Nt/2) / Nt );
%     end 
% end 
% 
% %% Constructing F
% F = zeros(Nc,Nc);
% 
% for i = 1 : Nc
%     for j = 1 : Nc
%         F(i,j) = exp(1i * 2 * pi * i * j / Nc );
%     end 
% end 
% 
% 
% %% Constructing Angle-delay profiles 
% 
% ADP = V' * H * F ;
% % fname = sprintf('test_%d.csv', iji);
% dlmwrite('test_final.csv',abs(ADP),'delimiter',',','-append','precision',3);
% % Angle_Delay_Profile((iiii-1) * N + K, 1, :, :) = abs(ADP);
% % Angle_Delay_Profile_LOS_Blocked(iiii,K,:,:) = ADP;
% % Angle_Delay_Profile_ANLOS(iiii,K,:,:) = ADP;
% 
% end
% end
% 
% % N_distortion = 30;
% % S_distortion = randi([1,300-N_distortion],1,1);
% % xbios=randi([floor(1*Nt/4) floor(3*Nt/4)],1,1);
% % ybios=randi([floor(1*Nt/4) floor(3*Nt/4)],1,1);
% 
% % for K = S_distortion : S_distortion + N_distortion
% %     Angle_Delay_Profile_LOS_Blocked(iiii,K,:,:) = LOS_Blockage(DeepMIMO_dataset{1}.user{P(K)}.channel,Nt,Nc);
% %     Angle_Delay_Profile_ANLOS(iiii,K,:,:) = ANLOS(DeepMIMO_dataset{1}.user{P(K)}.channel,Nt,Nc,xbios,ybios);
% % end
% % end
% 
% % fname = sprintf('DataADP_%d.mat', iii);
% % save(fname,'Channel','Location','Angle_Delay_Profile','Angle_Delay_Profile_LOS_Blocked','Angle_Delay_Profile_ANLOS')
% % end
%return
%}
%{
for iii = 1 : 181*1100/2
    iii
%     pause(5)
clearvars -except iii DeepMIMO_dataset iji idx THr

% DeepMIMO_dataset=DeepMIMO_Dataset_Generator;
% % N = 20;
% % W = 601; 
% % L = 181;
Nt = 64 ;  % Number of antennas at BS
Nc = 64 ;  % Number of Subcarriers 
% for iiii=1:25
% iiii
% P=RandomWalk2(N,W,L,DeepMIMO_dataset);

% for K=1:N
H = DeepMIMO_dataset{1}.user{(iii)}.channel;
L = DeepMIMO_dataset{1}.user{(iii)}.loc;
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
% fname = sprintf('test_%d.csv', iji);
dlmwrite('testforpython\fullADP_O13p5.csv',abs(ADP),'delimiter',',','-append','precision',4);
dlmwrite('testforpython\full_Loc_O13p5.csv',L,'delimiter',',','-append','precision',4);



% Angle_Delay_Profile((iii-1) * N + K, 1, :, :) = abs(ADP);
% 0Angle_Delay_Profile_LOS_Blocked(iiii,K,:,:) = ADP;
% Angle_Delay_Profile_ANLOS(iiii,K,:,:) = ADP;

end
for iii = 181*1100/2 + 1 : 181*1100
    iii
%     pause(5)
clearvars -except iii DeepMIMO_dataset iji idx THr

% DeepMIMO_dataset=DeepMIMO_Dataset_Generator;
% % N = 20;
% % W = 601; 
% % L = 181;
Nt = 64 ;  % Number of antennas at BS
Nc = 64 ;  % Number of Subcarriers 
% for iiii=1:25
% iiii
% P=RandomWalk2(N,W,L,DeepMIMO_dataset);

% for K=1:N
H = DeepMIMO_dataset{1}.user{(iii)}.channel;
L = DeepMIMO_dataset{1}.user{(iii)}.loc;
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
% fname = sprintf('test_%d.csv', iji);
dlmwrite('testforpython\fullADP1_O13p5.csv',abs(ADP),'delimiter',',','-append','precision',4);
dlmwrite('testforpython\full_Loc1_O13p5.csv',L,'delimiter',',','-append','precision',4);



% Angle_Delay_Profile((iii-1) * N + K, 1, :, :) = abs(ADP);
% Angle_Delay_Profile_LOS_Blocked(iiii,K,:,:) = ADP;
% Angle_Delay_Profile_ANLOS(iiii,K,:,:) = ADP;

end
%}