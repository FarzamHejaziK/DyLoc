% Random Walk  We have 8 possible move 1--8
function P=Randomwalk2(N,W,L,DeepMIMO_dataset)
%% Parameters 
Np = N;  %% number of walk
% starting point
P(1) = randi([1 L*W],1,1);
rw = randi([1 8],1,1);
% rw = 3;
Te(1) = rw;
for i = 2 : N
    Te(i) = rw;
    T = 0;
    while T == 0
        if rw == 1
            P(i) = P(i-1) - 1 + L;
            if  P(i) <= W*L & floor(P(i)/L) == floor(P(i-1)/L) + 1 
                T = 1;
                if norm(DeepMIMO_dataset{1}.user{P(i)}.loc-DeepMIMO_dataset{1}.user{P(i-1)}.loc,2) > 1
                T = 0;
                rw = 2;
                end
            else
                rw = 2;
            end
        end
        if rw == 2
            P(i) = P(i-1) - 1;
            if  P(i) > 0 & floor(P(i)/L) == floor(P(i-1)/L) 
                T = 1;
                if norm(DeepMIMO_dataset{1}.user{P(i)}.loc-DeepMIMO_dataset{1}.user{P(i-1)}.loc,2) > 1
                T = 0;
                rw = 3;
                end
            else
                rw = 3;
            end
        end
         if rw == 3
            P(i) = P(i-1) - L - 1;
            if  P(i) > 0 & floor(P(i)/L) == floor(P(i-1)/L)-1
                T = 1;
                if norm(DeepMIMO_dataset{1}.user{P(i)}.loc-DeepMIMO_dataset{1}.user{P(i-1)}.loc,2) > 1
                T = 0;
                rw = 4;
                end
               
            else
                rw = 4;
%                 DeepMIMO_dataset{1}.user{P(i-1)}.loc
%                 i
%                 DeepMIMO_dataset{1}.user{P(i-1)+L}.loc
            end
         end       
        if rw == 4
            P(i) = P(i-1) + L;
%             DeepMIMO_dataset{1}.user{P(i)}.loc;
            if  P(i) <= L * W & floor(P(i)/L) == floor(P(i-1)/L) + 1
                T = 1;
                if norm(DeepMIMO_dataset{1}.user{P(i)}.loc-DeepMIMO_dataset{1}.user{P(i-1)}.loc,2) > 1
                T = 0;
                rw = 4;
                end
            else
                rw = 5;
            end
        end
        if rw == 5
            P(i) = P(i-1) - L;
            if  P(i) > 0 & floor(P(i)/L) == floor(P(i-1)/L) - 1
                T = 1;
                if norm(DeepMIMO_dataset{1}.user{P(i)}.loc-DeepMIMO_dataset{1}.user{P(i-1)}.loc,2) > 1
                T = 0;
                rw = 6
                end    
            else
                rw = 6;
            end
        end
        if rw == 6
            P(i) = P(i-1) + L + 1; 
                if P(i) <= W * L & floor(P(i)/L) == floor(P(i-1)/L) + 1
                    T = 1;
                if norm(DeepMIMO_dataset{1}.user{P(i)}.loc-DeepMIMO_dataset{1}.user{P(i-1)}.loc,2) > 1
                T = 0;
                rw = 7;
                end
                else
                    rw = 7;
                end
        end
        if rw == 7
            P(i) = P(i-1) + 1;
                if P(i) <= W*L & floor(P(i)/L) == floor(P(i-1)/L)
                    T = 1;
                if norm(DeepMIMO_dataset{1}.user{P(i)}.loc-DeepMIMO_dataset{1}.user{P(i-1)}.loc,2) > 1
                T = 0;
                rw = 8;
                end
                else
                    rw = 8;
                end
        end
        if rw == 8
            P(i) = P(i-1) - L + 1;
                if P(i) > 0 & floor(P(i)/L) == floor(P(i-1)/L) -1 
                    T = 1;
                if norm(DeepMIMO_dataset{1}.user{P(i)}.loc-DeepMIMO_dataset{1}.user{P(i-1)}.loc,2) > 1
                T = 0;
                rw = 1;
                end    
                else
                    rw = 1;
                end
        end
    end
end
% for i = 1 : N
%     Loc(i,:) = DeepMIMO_dataset{1}.user{P(i)}.loc;
%     x(i) = Loc(i,1);
%     y(i) = Loc(i,2);
% end  
% figure
%  plot(x,y)
% end
% end
end

            
 

