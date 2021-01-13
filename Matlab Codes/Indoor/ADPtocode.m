% Detecting all paths, powers and locations, in ADP
function [Lloc,Power,Asmin,Asmax,Dsmin,Dsmax] = ADPtocode(H,Nt,Nc,Thr)
%% ADP 
% Constructing V
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

ADP = V' * H * F ;
Lm = imregionalmax(abs(ADP),8);
%% Cleaning
[A B]= (find(Lm == 1));
Npot = length(A);

for i = 1 : Npot
    if abs(ADP(A(i),B(i))) > abs(ADP(mod(A(i)-1,Nt)+(mod(A(i)-1,Nt)==0)*Nt,B(i))) & abs(ADP(A(i),B(i))) > abs(ADP(mod(A(i)+1,Nt)+(mod(A(i)+1,Nt)==0)*Nt,B(i))) & abs(ADP(A(i),B(i))) > abs(ADP(A(i),mod(B(i)-1,Nc)+(mod(B(i)-1,Nc)==0)*Nc)) & abs(ADP(A(i),B(i))) > abs(ADP(A(i),mod(B(i)+1,Nc)+(mod(B(i)+1,Nc)==0)*Nc))
        Lm(A(i),B(i))=1;
    else
        Lm(A(i),B(i))=0;
    end
end

[A B]= (find(Lm == 1));
Npot = length(A);
for i = 1 : Npot
    for j = 1 : Npot 
        if (A(i) == A(j)) & i ~=j
            if abs(B(i) - B(j)) <=4 || abs(B(i) - B(j)) >=Nc-3
                if abs(ADP(A(i),B(i))) > abs(ADP(A(j),B(j)))
                    Lm(A(j),B(j))=0;
                else
                    Lm(A(i),B(i))=0;
                end
            end 
        end
        if (B(i) == B(j)) & i ~= j
            if abs(A(i) - A(j)) <=4 || abs(A(i) - A(j)) >=Nt-3
                if abs(ADP(A(i),B(i))) > abs(ADP(A(j),B(j)))
                    Lm(A(j),B(j))=0;
                else
                    Lm(A(i),B(i))=0;
                end
            end 
        end
    end
end

MaxADP = max(max(abs(ADP)));
clear A B
[A B]= (find(Lm == 1));
Npot = length(A);

for i = 1 : Npot
    if abs(ADP(A(i),B(i))) <= MaxADP/30
        Lm(A(i),B(i))=0;
    end
end

%% Angle and Delay spread 

clear A B
[A B]= (find(Lm == 1));
Npot = length (A);

for i = 1 : Npot
    Power(i) = abs(ADP(A(i),B(i)));
end

for i = 1 : Npot
    Lloc(i,:)=[A(i) B(i)]';
end

for i = 1 : Npot
    t1 = 0;
    p1 = 1;
    Asmin(i) = A(i); 
    while t1 == 0
        if abs(ADP(Asmin(i),B(i))) <= Power(i)/Thr || p1 > 15
            t1 = 1;
        else
            Asmin(i) = mod(Asmin(i)-1,Nt)+(mod(Asmin(i)-1,Nt)==0)*Nt;
            p1 = p1 + 1;
        end
    end
    p2 = 1;
    t2 = 0;
    Asmax(i) = A(i); 
    while t2 == 0
        if abs(ADP(Asmax(i),B(i))) <= Power(i)/Thr || p2 > 15
            t2 = 1;
        else
            Asmax(i) = mod(Asmax(i)+1,Nt)+(mod(Asmax(i)+1,Nt)==0)*Nt;
            p2 = p2 +1;
        end
    end    
%     t3 = 0;
%     Asmax(i) = A(i); 
%     while t3 == 0
%         if abs(ADP(Asmax(i),B(i))) <= Power(i)/Thr
%             t1 = 1;
%         else
%             Asmax(i) = mod(Asmax(i)+1,Nt)+(mod(Asmax(i)+1,Nt)==0)*Nt;
%         end
%     end         
    t3 = 0;
    p3 = 1;
    Dsmin(i) = B(i); 
    while t3 == 0
        if abs(ADP(A(i),Dsmin(i))) <= Power(i)/Thr || p3 > 15
            t3 = 1;
        else
            Dsmin(i) = mod(Dsmin(i)-1,Nc)+(mod(Dsmin(i)-1,Nc)==0)*Nc;
            p3 = p3+1;
        end
    end          
    t4 = 0;
    p4 = 1;
    Dsmax(i) = B(i); 
    while t4 == 0
        if abs(ADP(A(i),Dsmax(i))) <= Power(i)/Thr || p4 > 15
            t4 = 1;
        else
            Dsmax(i) = mod(Dsmax(i)+1,Nc)+(mod(Dsmax(i)+1,Nc)==0)*Nc;
            p4 = p4 + 1;
        end
    end   
end

end
    
