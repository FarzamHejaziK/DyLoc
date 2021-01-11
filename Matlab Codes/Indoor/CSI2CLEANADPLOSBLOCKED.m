function [ADP] = CSI2CLEANADPLOSBLOCKED(H,Nt,Nc,Thr1,Thr2)
%% CSI to Code
[Lloc,Power,Asmin,Asmax,Dsmin,Dsmax] = ADPtocode(H,Nt,Nc,Thr1);
ADP = abs(CSI2ADP(H,Nt,Nc));
[Lloc,Power,Asmin,Asmax,Dsmin,Dsmax] = ADPtocode(H,Nt,Nc,Thr2);
a=find(Power==max(Power));

for i = 1 : Nt
    for j = 1 : Nc
        p = zeros(length(Power),1);
        for ii = 1 : length(Power)
            if Asmin(ii) > Asmax(ii)
                Asmax(ii) = Asmax(ii) + Nt;
                t1 = 1; 
            end
            if Dsmin(ii) > Dsmax(ii)
                Dsmax(ii) = Dsmax(ii) + Nc;
            end
            if ((Asmin(ii) < i && Asmax(ii) > i) || (Asmin(ii) < i+Nt && Asmax(ii) > i+Nt)) && ((Dsmin(ii) < j && Dsmax(ii) > j) || (Dsmin(ii) < j+Nc && Dsmax(ii) > j+Nc))
                p(ii) = 1;
            else
                p(ii) = 0;
            end
        end
        if sum(p) == 0
            ADP(i,j) = 0;
        end 
    end
end

for i = 1 : Nt
    for j = 1 : Nc
            if Asmin(a) > Asmax(a)
                Asmax(a) = Asmax(a) + Nt;
                t1 = 1; 
            end
            if Dsmin(a) > Dsmax(a)
                Dsmax(a) = Dsmax(ii) + Nc;
            end
            if ((Asmin(a) < i && Asmax(a) > i) || (Asmin(a) < i+Nt && Asmax(a) > i+Nt)) && ((Dsmin(a) < j && Dsmax(a) > j) || (Dsmin(a) < j+Nc && Dsmax(a) > j+Nc))
                ADP(i,j) = 0;
            end
    end
end
% figure(1)
% imagesc(abs(CSI2ADP(H,Nt,Nc)))
% figure(2)
% imagesc(ADP)
end