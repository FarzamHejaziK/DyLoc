
format short g


%% Generating training dataset
for iii = 1 : 10000
    iii

clearvars -except iii DeepMIMO_dataset

% DeepMIMO_dataset=DeepMIMO_Dataset_Generator;
N = 20;  % Number of Frames
W = 110; % length of the test Dataset, e.g. from R100 to R200
L = 121; % length of each rows (from DeepMIMO website)
Nt = 32 ;  % Number of antennas at BS
Nc = 32 ;  % Number of Subcarriers 

P=RandomWalk2(N,W,L,DeepMIMO_dataset);

for K=1:N
H = DeepMIMO_dataset{1}.user{P(K)}.channel;
L = DeepMIMO_dataset{1}.user{P(K)}.loc;

 

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
dlmwrite('Moving_ADP_I3.csv',abs(ADP),'delimiter',',','-append','precision',4);


end


%% Generating Testing dataset

for iii = 1 : 2000
    iii

clearvars -except iii DeepMIMO_dataset

% DeepMIMO_dataset=DeepMIMO_Dataset_Generator;
N = 20;
W = 110; 
L = 121;
Nt = 32 ;  % Number of antennas at BS
Nc = 32 ;  % Number of Subcarriers 
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
dlmwrite('Moving_ADP_test_I3.csv',abs(ADP),'delimiter',',','-append','precision',4);


end



end
