function [ADP] = CSI2CLEANADP(H,Nt,Nc,Thr)
%% CSI to Code
[Lloc,Power,Asmin,Asmax,Dsmin,Dsmax] = ADPtocode(H,Nt,Nc,Thr);
ADP = abs(CSI2ADP(H,Nt,Nc));
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
end
              
    