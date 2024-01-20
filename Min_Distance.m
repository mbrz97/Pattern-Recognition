clc
clear

for z = 1:30
    
% Loading data set
load('aviris.mat');
idp1_reshape = reshape(idp1, [145*145,190]);
thematic_reshape = reshape(thematic, [145*145,1]);

% PCA
n = 80;
[coeff, score, latent, tsquared, explained, mu] = pca(idp1_reshape, 'NumComponents', n);


data1 = score(:, 1:n);
data = zeros(145*145, n+1);
data(:, 1) = thematic_reshape;
data(:, 2:end) = data1;
datasort = sortrows(data, 1);

%mu = mean( data(:, 2:191));
data_rebuild = score(:, 1:n) * coeff(:, 1:n).' + mu;

% Initializing train and test matrices
train = zeros( 14716, n+1);
test = zeros( 6308, n+1);


 
% Find indices of each class
class0 = find(datasort(:, 1) == 0);
class1 = find(datasort(:, 1) == 1);
class2 = find(datasort(:, 1) == 2);
class3 = find(datasort(:, 1) == 3);
class4 = find(datasort(:, 1) == 4);
class5 = find(datasort(:, 1) == 5);
class6 = find(datasort(:, 1) == 6);
class7 = find(datasort(:, 1) == 7);
class8 = find(datasort(:, 1) == 8);
class9 = find(datasort(:, 1) == 9);
class10 = find(datasort(:, 1) == 10);
class11 = find(datasort(:, 1) == 11);
class12 = find(datasort(:, 1) == 12);
class13 = find(datasort(:, 1) == 13);
class14 = find(datasort(:, 1) == 14);
class15 = find(datasort(:, 1) == 15);
class16 = find(datasort(:, 1) == 16);

% Creating a matrix for holding start and finish indices of each class
% Row 1 = start positions & Row 2 = end positions
% Each column(1-17) represents a class(0-16)
class = [class0(1,1), class1(1,1), class2(1,1), class3(1,1),...
         class4(1,1), class5(1,1), class6(1,1), class7(1,1),...
         class8(1,1), class9(1,1), class10(1,1), class11(1,1),...
         class12(1,1), class13(1,1), class14(1,1),...
         class15(1,1),  class16(1,1);
         class0(end), class1(end), class2(end), class3(end),...
         class4(end), class5(end), class6(end), class7(end),...
         class8(end), class9(end), class10(end), class11(end),...
         class12(end), class13(end), class14(end),...
         class15(end),class16(end)];

% Initializing pointers for knowing the last appended positions
pointer = 0;
testpointer = 0;

% Creating a Matrix for holding start and finish of each class in Train
train_pos = zeros(2, 17);
test_pos = zeros(2, 17);

% Creating train and test matrices with random data
for i = 1:17
    if i == 1
        shuffle = randperm(fix(class(2,i)));
        train(1 : pointer + length(randperm(fix(0.7*class(2,i)))) , 1:n+1) =...
            datasort( shuffle(1:fix(0.7*length(shuffle))) , 1:n+1);
    
        test(1: testpointer + length(randperm(round(0.3*class(2,i)))) , 1:n+1) = ...
            datasort( shuffle(fix(0.7*length(shuffle))+1:end) , 1:n+1);
        
        train_pos(1,i) = 1;
        train_pos(2,i) = pointer + length(randperm(fix(0.7*class(2,i))));
        test_pos(1,i) = 1;
        test_pos(2,i) = testpointer + length(randperm(fix(0.3*class(2,i))));
        testpointer = length(randperm(fix(0.3*class(2,i)))) + 1;
        pointer = length(randperm(fix(0.7*class(2,i)))) + 1;
    else
        shuffle = randperm(fix(class(2,i)-class(2,i-1)));
        train(pointer : pointer-1 + length(randperm(fix(0.7*(class(2,i) - class(2,i-1))))) , 1:n+1) =...
            datasort(class(2,i-1)+shuffle(1:fix(0.7*length(shuffle))), 1:n+1);
        
         test(testpointer: testpointer-1 + length(shuffle(fix(0.7*length(shuffle))+1:end)) , 1:n+1) = ...
            datasort( class(2,i-1)+shuffle(fix(0.7*length(shuffle))+1:end) , 1:n+1);
        
        train_pos(1,i) = pointer;
        train_pos(2,i) = pointer-1 + length(randperm(fix(0.7*(class(2,i) - class(2,i-1)))));
        test_pos(1,i) = testpointer;
        test_pos(2,i) = testpointer-1 + length(shuffle(round(0.7*length(shuffle))+1:end));
        testpointer = length(randperm(fix(0.3*class(2,i)))) + 1;
        pointer = length(randperm(fix(0.7*class(2,i)))) + 1;
    end
end

% Calculating mean for each class, using the train matrix
means = zeros(17,n);

% Calcuating mean of each class
for i = 1:17
    means(i, :) = mean( train( (train_pos(1,i):train_pos(2,i)), 2:n+1) );
end

labels = zeros(17, 3197);

% Calculating minimum distance for each data
for i =1
%for i = 1:17
    pos = 0;
    %for j = test_pos(1,i): (test_pos(2,i))
    for j = 1:145*145
        pos = pos+1;
        mins = zeros(1,17);
        for k = 1:17
            %d = (sum((means(k,:) - test(j, 2:n+1)).^2))^0.5;
            d = (sum((means(k,:) - data(j, 2:n+1)).^2))^0.5;
            mins(1, k) = d;
        end
        [val, labels(i, pos)] = min(mins);
    end
end

% Accuracy for each class

for i = 1:17
    acc(z,i) = (sum(labels(i, 1:(1+test_pos(2,i)-test_pos(1,i)))==i))/(1+test_pos(2,i)-test_pos(1,i)) * 100;
end

end


% Accuracy mean
acc_mean = mean(acc);
acc_std = std(acc);

%Accuracy mean for all
% x = 0;
% for i = 1:145*145
%    if(labels(1,i) == data(i,1)+1)
%        x = x+1;
%    end
% end
% 
% all_acc = ( x/ 21025 )*100;