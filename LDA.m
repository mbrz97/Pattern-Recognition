clc
clear


    
% Loading data set
load('aviris.mat');
idp1_reshape = reshape(idp1, [145*145,190]);
thematic_reshape = reshape(thematic, [145*145,1]);

% PCA
n = 16;


%data1 = score(:, 1:n);
data = zeros(145*145, 191);
data(:, 1) = thematic_reshape;
data(:, 2:end) = idp1_reshape;
datasort = sortrows(data, 1);


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




% Train Percentage
TR = 0.7;

% Test Percentage
TS = 1 - TR;

train = zeros( fix(TR*(145*145)), 17);
test = zeros( fix(TS*(145*145)), 17);

% Creating a Matrix for holding start and finish of each class in Train
train_pos = zeros(2, 16);
test_pos = zeros(2, 16);

pointer = 0;
testpointer = 0;
for i = 1:17
    if i == 1
        shuffle = randperm(fix(class(2,i)));
        train(1 : pointer + length(randperm(fix(TR*class(2,i)))) , 1:n+1) =...
            datasort( shuffle(1:fix(TR*length(shuffle))) , 1:n+1);
    
%        test(1: testpointer + length(randperm(fix(TS*class(2,i)))) , 1:n+1) = ...
            datasort( shuffle(fix(TR*length(shuffle))+1:end) , 1:n+1);
        
        train_pos(1,i) = 1;
        train_pos(2,i) = pointer + length(randperm(fix(TR*class(2,i))));
        test_pos(1,i) = 1;
        test_pos(2,i) = testpointer + length(randperm(fix(TS*class(2,i))));
        testpointer = length(randperm(fix(TS*class(2,i)))) + 1;
        pointer = length(randperm(fix(TR*class(2,i)))) + 1;
    else
        shuffle = randperm(fix(class(2,i)-class(2,i-1)));
        train(pointer : pointer-1 + length(randperm(fix(TR*(class(2,i) - class(2,i-1))))) , 1:n+1) =...
            datasort(class(2,i-1)+shuffle(1:fix(TR*length(shuffle))), 1:n+1);
        
         test(testpointer: testpointer-1 + length(shuffle(fix(TR*length(shuffle))+1:end)) , 1:n+1) = ...
            datasort( class(2,i-1)+shuffle(fix(TR*length(shuffle))+1:end) , 1:n+1);
        
        train_pos(1,i) = pointer;
        train_pos(2,i) = pointer-1 + length(randperm(fix(TR*(class(2,i) - class(2,i-1)))));
        test_pos(1,i) = testpointer;
        test_pos(2,i) = testpointer-1 + length(shuffle(round(TR*length(shuffle))+1:end));
        testpointer = length(randperm(fix(TS*class(2,i)))) + 1;
        pointer = length(randperm(fix(TR*class(2,i)))) + 1;
    end
end









     
%%%%% LDA 


% Calculating mean for each class, using the train matrix
means = zeros(16,190);

Sizes = zeros(1,17);
for i = 1:17
   Sizes(1, i) =  train(2,i) - train(1,i) + 1;
end

trainsort = zeros(fix(TR*145*145), 191);
trainsort(:, 1) = train(:,1);
trainsort(:, 2:end) = train(:,2:191);
trainsort = sortrows(trainsort, 1);

% Calcuating mean of each class
for i = 1:16
    means(i, 1:190) = mean( datasort( (class(1,i+1):class(1,i+1)+fix(0.7*Sizes(1,i+1))), 2:191) );
end

covAll = zeros(190, 190, 16);
% Calculating Covariance of each class
for i = 1:16
   covAll(:, :, i) = cov( datasort( (class(1,i+1):class(1,i+1)+fix(0.7*Sizes(1,i+1))), 2:191) );
end

meanTotal = zeros(1, 190);
meanTotal = mean( datasort(:, 2:191) );





SB=0;
SW=zeros(190,190);
Si=0;


% for i = 1:17
%     for j = 0:Sizes(i)-1
%         Si = Si + (datasort(class(1,i)+j, 2:191) - means(i,:)).'*(datasort(class(1,i)+j, 2:191) - means(i,:)) ;
%     end
%     SW = SW + Si;
% end


 for i =1:16
     SB=SB+Sizes(i).*(means(i,:)-meanTotal).'*(means(i,:)-meanTotal);
     SW = SW + Sizes(1, i).*(covAll(:,:, i)+0.01*eye(190));
 end
 
A=inv(SW)*SB;
[V,D] = eig(A);
eigsort=sort(V,'descend');
newT=train(:,2:191)*V(:,1:16);
data1 = zeros(145*145, 17);
data1(:, 1) = datasort(:, 1);
data1(:, 2:end) = newX;
datasort = sortrows(data1, 1);

%%%%%%%%%%%%%%%%%%%% LDA










labels = zeros(1, 145*145);


% Calculating mean for each class, using the train matrix
meanT = zeros(16,n);

% Calcuating mean of each class
for i = 1:16
    meanT(i, :) = mean( train( (train_pos(1,i+1):train_pos(2,i+1)), 2:n+1) );
end


% Covariance calculation and inverse
covclass = zeros(n, n, 16);
invcov = zeros(n, n, 16);
logdet = zeros(1, 16);

% Calculating covariances for each class
for i = 1:16
    % Adding 0.01*eye(190) because of matrix singularity error
    covclass(:, :, i) = cov(train(train_pos(1, i+1):train_pos(2,i+1), 2:n+1)) + 0.01.*eye(n);
    invcov(:, :, i) = inv(covclass(:, :, i));
    logdet(1, i) = log(det(covclass(:, :, i)));
end


 
% Calculating minimum distance for each data

    pos = 0;
    for j = 1:145*145
        pos = pos+1;
        if thematic_reshape(j) == 0
            continue
        end
        mins = zeros(1,16);
        for k = 1:16
            %d = (test(j, 2:n+1) - means(k,:))*invcov(:, :, k)*(test(j, 2:n+1) - means(k,:))'+ logdet(1, k);
            d = (newX(j, 1:16) - meanT(k,:))*invcov(:, :, k)*(newX(j, 1:16) - meanT(k,:))'+ logdet(1, k);
            mins(1, k) = d;
        end
        [val, labels(1, pos)] = min(mins);
    end

 %Accuracy for each class
 summ = 0;
for i = 1:145*145
    if data(i, 1) == 0 
        labels(1, i) = 0;
    else
        if labels(1,i) == data(i, 1)
            summ =summ +1;
        end
    end
end

acc(1,1) = summ / 10365 * 100;


% Accuracy mean
%acc_mean = mean(acc);
%acc_std = std(acc);
%Accuracy mean for all
% x = 0;
% for i = 1:145*145
%    if(labels(1,i) == data(i,1)+1)
%        x = x+1;
%    end
% 
% all_acc = ( x/ 21025 )*100;

figure; imagesc( reshape( labels(1,:), [145,145] ) )
figure; imshow( reshape( newX(:, 1), [145,145] ), [] )
figure; imagesc(thematic )

