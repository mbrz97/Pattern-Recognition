clc
clear

% Transistion Matrix
PS =[1/3, 1/3, 1/3 ;
     1/3, 1/3, 1/3 ;
     1/3, 1/3, 1/3];

% Head Possibility
PH = [0.5, 0.75, 0.25];

% Tail Possibility
PT = ones(1, 3) - PH;

% Generate Random State
Start_State = randi(3, 1);

seq = zeros(100, 1);
coin = zeros(100, 1);
State = Start_State;

% Head Count initialization
H = 0;

% Tails Count initialization
T= 0;

for i = 1:100
        
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
