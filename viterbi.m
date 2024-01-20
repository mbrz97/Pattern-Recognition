clc
clear

N = 10000;

% Initial probability distribution of the state
pii = [1/3, 1/3, 1/3];

% Transistion Matrix
A =[0.25, 0.5, 0.25 ;
     0.4, 0.2, 0.4 ;
     0.3, 0.5, 0.2];

% Head Possibility (1)
PH = [0.1, 0.8, 0.4];

% Tail Possibility (0)
PT = ones(1, 3) - PH;

% State Space
S = [1, 2, 3];

% Observation Space
O = [0 ,1];


% Emission Matrix  (NUMBER OF STATES)*(HEAD OR TAIL)   (i = 3* j = 2)
%  the probability of observing Oj from state Si
B = [PT(1) ,PH(1);
     PT(2) ,PH(2);
     PT(3) ,PH(3)];


% Generate Random State
Start_State = randi(3, 1);

% Sequence of States
seq = zeros(1, N);

% Real Sequence of Observations
coin = zeros(1, N);


State = Start_State;

% Head Count initialization
H = 0;

% Tails Count initialization
T= 0;

% Delta initialization (Num of States)*(Num of Observations)
delta = zeros(3, N);


% Psi initialization
psi = zeros(3, N);


for i = 1:N
        
        p = rand;
        
        if p <= PH(State)    % Count as heads
            coinS = 1;
            H = H + 1;
        else                % Count as tails
            coinS = 0;
            T = T + 1;
        end
        
        % Save coin state
        coin(i) = coinS;
        
        % Probability of next state
        next_p = rand;
        
        if next_p <= 1/3
            State = 1;
        elseif next_p <= 2/3
            State = 2;
        elseif next_p <= 1
            State = 3;
        end
        
        % Save selected coin
        seq(i) = State;
end

delta(:, 1) = pii(1, :)'.*B(:,coin(1,1)+1);

for i = 2:N
    for j = 1:3
        delta(j, i) = max( delta(:, i-1).*A(:, j).*B(j,coin(i)+1) );
        [val, psi(j, i)] = max( delta(:, i-1).*A(:, j) );
    end
end

% Define most probable states q*
qstar = zeros(1, N);
[qVal, qstar(1, N)] = max(delta(:,N));

for i = N-1:-1:1
    qstar(1, i) = psi( qstar(1, i+1) , i+1);
end


% Accuracy
r = 0;
for i = 1:N
   if seq(1,i) == qstar(1,i)
       r = r+1;
   end
end
r = r*100/N;
