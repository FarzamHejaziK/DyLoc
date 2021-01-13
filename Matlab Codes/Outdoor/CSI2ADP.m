%% Converting CSI to ADP

function [ADP] = CSI2ADP(H,Nt,Nc)
%%
V = zeros(Nt,Nt);
for i = 1 : Nt
    for j = 1 : Nt
        V(i,j) = exp(-1i * 2 * pi * i * ( j - Nt/2) / Nt );
    end 
end 

% Constructing F
F = zeros(Nc,Nc);

for i = 1 : Nc
    for j = 1 : Nc
        F(i,j) = exp(1i * 2 * pi * i * j / Nc );
    end 
end 

ADP = V' * (H) * F ;
end