clc
clear

% NEXT SESSION: 
% TAKE 60% TRAIN 40% TEST

for z = 1:1
    
% Loading data set
load('aviris.mat');
idp1_reshape = reshape(idp1, [145*145,190]);
thematic_reshape = reshape(thematic, [145*145,1]);

% PCA
n = 190;
%[coeff, score, latent, tsquared, explained, mu] = pca(idp1_reshape, 'NumComponents', n);


%data1 = score(:, 1:n);
data = zeros(145*145, 191);
data(:, 1) = thematic_reshape;
data(:, 2:end) = idp1_reshape;
datasort = sortrows(data, 1);

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate eigenvalues and eigenvectors of the covariance matrix
% X1 = idp1_reshape;
% X1 = X1 - mean(X1);
% covarianceMatrix = cov(X1);
% [V,D] = eig(covarianceMatrix);
% dataInPrincipalComponentSpace = X1*coeff;

% The columns of X*coeff are orthogonal to each other.
% This is shown with ...
% corrcoef = corrcoef(dataInPrincipalComponentSpace);

% The variances of these vectors are the eigenvalues of the covariance matrix,
% and are also the output "latent". Compare these three outputs
% dataInPCA = var(dataInPrincipalComponentSpace)';

% Latent
% sortDiag = sort(diag(D),'descend');


% "coeff" are the principal component vectors.
% These are the eigenvectors of the covariance matrix.
% Compare "coeff" and "V". Notice that they are the same


%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%




%mu = mean( data(:, 2:191));
% data_rebuild = score(:, 1:n) * coeff(:, 1:n).' + mu;

% Initializing train and test matrices

% Train Percentage
TR = 0.7;

% Test Percentage
TS = 1 - TR;

train = zeros( fix(TR*(145*145 - 10659)), 190);
test = zeros( fix(TS*(145*145 - 10659)), 190);

 
% Find indices of each class
%class0 = find(datasort(:, 1) == 0);
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
class = [class1(1,1), class2(1,1), class3(1,1),...
         class4(1,1), class5(1,1), class6(1,1), class7(1,1),...
         class8(1,1), class9(1,1), class10(1,1), class11(1,1),...
         class12(1,1), class13(1,1), class14(1,1),...
         class15(1,1),  class16(1,1);
         class1(end), class2(end), class3(end),...
         class4(end), class5(end), class6(end), class7(end),...
         class8(end), class9(end), class10(end), class11(end),...
         class12(end), class13(end), class14(end),...
         class15(end),class16(end)];

% Initializing pointers for knowing the last appended positions
pointer = 1;
testpointer = 1;

% Creating a Matrix for holding start and finish of each class in Train
train_pos = zeros(2, 16);
test_pos = zeros(2, 16);

% Creating train and test matrices with random data
for i = 1:16
%     if i == 1
%         shuffle = randperm(fix(class(2,i)));
%         train(1 : pointer + length(randperm(fix(TR*class(2,i)))) , 1:n+1) =...
%             datasort( shuffle(1:fix(TR*length(shuffle))) , 1:n+1);
%     
% %        test(1: testpointer + length(randperm(fix(TS*class(2,i)))) , 1:n+1) = ...
%             datasort( shuffle(fix(TR*length(shuffle))+1:end) , 1:n+1);
%         
%         train_pos(1,i) = 1;
%         train_pos(2,i) = pointer + length(randperm(fix(TR*class(2,i))));
%         test_pos(1,i) = 1;
%         test_pos(2,i) = testpointer + length(randperm(fix(TS*class(2,i))));
%         testpointer = length(randperm(fix(TS*class(2,i)))) + 1;
%         pointer = length(randperm(fix(TR*class(2,i)))) + 1;
%     else
        shuffle = randperm(fix(class(2,i)-class(1,i)));
        train(pointer : pointer-1 + length(randperm(fix(TR*(class(2,i) - class(1,i))))) , 1:n+1) =...
            datasort(class(1,i)+shuffle(1:fix(TR*length(shuffle))), 1:n+1);
        
         %test(testpointer: testpointer-1 + length(shuffle(fix(TR*length(shuffle))+1:end)) , 1:n+1) = ...
         %  datasort( class(1,i)+shuffle(fix(TR*length(shuffle))+1:end) , 1:n+1);
        
        train_pos(1,i) = pointer;
        train_pos(2,i) = pointer-1 + length(randperm(fix(TR*(class(2,i) - class(1,i) + 1))));
        test_pos(1,i) = testpointer;
        test_pos(2,i) = testpointer-1 + length(shuffle(round(TR*length(shuffle))+1:end));
        testpointer = length(randperm(fix(TS*class(2,i)))) + 1;
        pointer = pointer + length(randperm(fix(TR*(class(2,i) - class(1,i) + 1))));
    
end

% Calculating mean for each class, using the train matrix
means = zeros(16,n);

% Calcuating mean of each class
for i = 1:16
    means(i, :) = mean( train( (train_pos(1,i):train_pos(2,i)), 2:n+1) );
end

labels = zeros(16, fix(TS*145*145));




% Covariance calculation and inverse
covclass = zeros(n, n, 16);
invcov = zeros(n, n, 16);
logdet = zeros(1, 16);

% Calculating covariances for each class
for i = 1:16
    % Adding 0.01*eye(190) because of matrix singularity error
    covclass(:, :, i) = cov(train(train_pos(1, i):train_pos(2,i), 2:n+1)) + 0.01.*eye(n);
    invcov(:, :, i) = inv(covclass(:, :, i));
    logdet(1, i) = log(det(covclass(:, :, i)));
end

% Mean of all Train data
meantrain = zeros(1,190);
meantrain = mean(train(:, 2:191));


% Size of each class
% Sizes = [size(class1, 1), size(class2, 1), size(class3, 1), size(class4, 1), size(class5, 1), size(class6, 1), ...
%         size(class7, 1), size(class8, 1), size(class9, 1), size(class10, 1), size(class11, 1), size(class12, 1), ...
%         size(class13, 1), size(class14, 1), size(class15, 1), size(class16, 1)];

Sizes = zeros(1, 16);
for i = 1:16
   Sizes(1, i) =  train_pos(2,i) - train_pos(1,i) + 1;
end

vars = zeros(16,190);
for i = 1:16
    vars(i, :) = var(train(train_pos(1,i):train_pos(2,i), 2:191 ));
end


SB=0;
SW=0;
Si=0;
for i = 1:16
    for j = 0:Sizes(i)
        Si = Si + (train(train_pos(1,i)+j, 2:191) - means(i,:)).'*(train(train_pos(1,i)+j, 2:191) - means(i,:)) ;
    end
    %Si = Sizes(1,i).*vars(i, :);
    SW = SW + Si;
end

 for i =1:16
     SB=SB+Sizes(i).*(means(i,:)-meantrain).'*(means(i,:)-meantrain);
 end

 ST = SB + SW;
 
A=SB*inv(SW);
[V,D] = eig(A);
newX=train2(:,1:190)*V(:,1:16);

 
% Calculating minimum distance for each data
for i = 1
%for i = 1:17
    pos = 0;
    %for j = test_pos(1,i): (test_pos(2,i))
    for j = 1:145*145
        pos = pos+1;
        mins = zeros(1,16);
        for k = 2:17
            %d = (test(j, 2:n+1) - means(k,:))*invcov(:, :, k)*(test(j, 2:n+1) - means(k,:))'+ logdet(1, k);
            d = (data(j, 2:n+1) - means(k,:))*invcov(:, :, k)*(data(j, 2:n+1) - means(k,:))'+ logdet(1, k);
            mins(1, k) = d;
        end
        [val, labels(i, pos)] = min(mins(2:17));
    end
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

acc(1,z) = summ / 10365 * 100;

end

% Accuracy mean
%acc_mean = mean(acc);
%acc_std = std(acc);
%Accuracy mean for all
% x = 0;
% for i = 1:145*145
%    if(labels(1,i) == data(i,1)+1)
%        x = x+1;
%    end
% end
% 
% all_acc = ( x/ 21025 )*100;

figure; imagesc( reshape( labels(1,:), [145,145] ) )
figure; imshow( reshape( score(:, 1), [145,145] ), [] )
