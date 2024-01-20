clc
clear

for z = 1:1
    
% Loading data set
load('aviris.mat');
idp1_reshape = reshape(idp1, [145*145,190]);
thematic_reshape = reshape(thematic, [145*145,1]);

% Concatinating idp1 and thematic
data = zeros(145*145, 191);
data(:, 1) = thematic_reshape;
data(:, 2:end) = idp1_reshape;

% Initializing train and test matrices
train = zeros( 14716, 191);
test = zeros( 6308, 191);

% Sorting all data based on class (0-16)
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

s{1} = find(data(:, 1) == 0);
s{2} = find(data(:, 1) == 1);
s{3} = find(data(:, 1) == 2);
s{4} = find(data(:, 1) == 3);
s{5} = find(data(:, 1) == 4);
s{6} = find(data(:, 1) == 5);
s{7} = find(data(:, 1) == 6);
s{8} = find(data(:, 1) == 7);
s{9} = find(data(:, 1) == 8);
s{10} = find(data(:, 1) == 9);
s{11} = find(data(:, 1) == 10);
s{12} = find(data(:, 1) == 11);
s{13} = find(data(:, 1) == 12);
s{14} = find(data(:, 1) == 13);
s{15} = find(data(:, 1) == 14);
s{16} = find(data(:, 1) == 15);
s{17} = find(data(:, 1) == 16);



cp1 = mvnrnd(mean(datasort(class1(:,1),2:191)),cov(datasort(class1(:,1),2:191)),200);
cp4 = mvnrnd(mean(datasort(class1(:,1),2:191)),cov(datasort(class1(:,1),2:191)),200);

cp7 = mvnrnd(mean(datasort(class1(:,1),2:191)),cov(datasort(class1(:,1),2:191)),200);
cp9 = mvnrnd(mean(datasort(class1(:,1),2:191)),cov(datasort(class1(:,1),2:191)),200);
cp13 = mvnrnd(mean(datasort(class1(:,1),2:191)),cov(datasort(class1(:,1),2:191)),200);

cp16 = mvnrnd(mean(datasort(class1(:,1),2:191)),cov(datasort(class1(:,1),2:191)),200);

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
        train(1 : pointer + length(randperm(fix(0.7*class(2,i)))) , 1:191) =...
            datasort( shuffle(1:fix(0.7*length(shuffle))) , 1:191);
    
        test(1: testpointer + length(randperm(round(0.3*class(2,i)))) , 1:191) = ...
            datasort( shuffle(fix(0.7*length(shuffle))+1:end) , 1:191);
        
        train_pos(1,i) = 1;
        train_pos(2,i) = pointer + length(randperm(fix(0.7*class(2,i))));
        test_pos(1,i) = 1;
        test_pos(2,i) = testpointer + length(randperm(fix(0.3*class(2,i))));
        testpointer = length(randperm(fix(0.3*class(2,i)))) + 1;
        pointer = length(randperm(fix(0.7*class(2,i)))) + 1;
    else
        shuffle = randperm(fix(class(2,i)-class(2,i-1)));
        train(pointer : pointer-1 + length(randperm(fix(0.7*(class(2,i) - class(2,i-1))))) , 1:191) =...
            datasort(class(2,i-1)+shuffle(1:fix(0.7*length(shuffle))), 1:191);
        
         test(testpointer: testpointer-1 + length(shuffle(fix(0.7*length(shuffle))+1:end)) , 1:191) = ...
            datasort( class(2,i-1)+shuffle(fix(0.7*length(shuffle))+1:end) , 1:191);
        
        train_pos(1,i) = pointer;
        train_pos(2,i) = pointer-1 + length(randperm(fix(0.7*(class(2,i) - class(2,i-1)))));
        test_pos(1,i) = testpointer;
        test_pos(2,i) = testpointer-1 + length(shuffle(round(0.7*length(shuffle))+1:end));
        testpointer = length(randperm(fix(0.3*class(2,i)))) + 1;
        pointer = length(randperm(fix(0.7*class(2,i)))) + 1;
    end
end

x = 0.5;
c0 =(train(train_pos(1,1):train_pos(2,1),2:191));
c1 =(train(train_pos(1,2):train_pos(2,2),2:191));
c2 =(train(train_pos(1,3):train_pos(2,3),2:191));
c3 =(train(train_pos(1,4):train_pos(2,4),2:191));
c4 =(train(train_pos(1,5):train_pos(2,5),2:191));
c5 =(train(train_pos(1,6):train_pos(2,6),2:191));
c6 =(train(train_pos(1,7):train_pos(2,7),2:191));
c7 =(train(train_pos(1,8):train_pos(2,8),2:191));
c8 =(train(train_pos(1,9):train_pos(2,9),2:191));
c9 =(train(train_pos(1,10):train_pos(2,10),2:191));
c10 =(train(train_pos(1,11):train_pos(2,11),2:191));
c11 =(train(train_pos(1,12):train_pos(2,12),2:191));
c12 =(train(train_pos(1,13):train_pos(2,13),2:191));
c13 =(train(train_pos(1,14):train_pos(2,14),2:191));
c14 =(train(train_pos(1,15):train_pos(2,15),2:191));
c15 =(train(train_pos(1,16):train_pos(2,16),2:191));
c16 =(train(train_pos(1,17):train_pos(2,17),2:191));

%GMM
GMModel{1} = fitgmdist(c0,1,'RegularizationValue',0.1,'Start','plus');
GMModel{2} = fitgmdist([c1; cp1],1,'RegularizationValue',0.1,'Start','plus');
GMModel{3} = fitgmdist(c2,2,'RegularizationValue',0.1,'Start','plus');
GMModel{4} = fitgmdist(c3,2,'RegularizationValue',0.1,'Start','plus');
GMModel{5} = fitgmdist([c4; cp4],2,'RegularizationValue',0.1,'Start','plus');
GMModel{6} = fitgmdist(c5,2,'RegularizationValue',0.1,'Start','plus');
GMModel{7} = fitgmdist(c6,2,'RegularizationValue',0.1,'Start','plus');
GMModel{8} = fitgmdist([c7; cp7],1,'RegularizationValue',0.1,'Start','plus');
GMModel{9} = fitgmdist(c8,1,'RegularizationValue',0.1,'Start','plus');
GMModel{10} = fitgmdist([c9; cp9],1,'RegularizationValue',0.1,'Start','plus');
GMModel{11} = fitgmdist(c10,4,'RegularizationValue',0.1,'Start','plus');
GMModel{12} = fitgmdist(c11,5,'RegularizationValue',0.1,'Start','plus');
GMModel{13} = fitgmdist(c12,3,'RegularizationValue',0.1,'Start','plus');
GMModel{14} = fitgmdist([c13; cp13],1,'RegularizationValue',0.1,'Start','plus');
GMModel{15} = fitgmdist(c14,3,'RegularizationValue',0.1,'Start','plus');
GMModel{16} = fitgmdist(c15,1,'RegularizationValue',0.1,'Start','plus');
GMModel{17} = fitgmdist([c16; cp16],1,'RegularizationValue',0.1,'Start','plus');

for i = 1
    pos = 0;
    for j = 1:145*145
        pos = pos+1;
        maxim = zeros(1,17);
        
        for k = 1:17
           mPdf = pdf(GMModel{k}, data(j, 2:191));
           maxim(1, k) = mPdf;
        end

        [val, labels(i, pos)] = max(maxim);
    end
end


% Accuracy for each class
correct = 0;
for i = 1:145*145
    newlabels = labels - 1;
    %acc(z,i) = (sum(labels(i, 1:(1+test_pos(2,i)-test_pos(1,i)))==i))/(1+test_pos(2,i)-test_pos(1,i)) * 100;
    if newlabels(1, i) == data(i, 1)
        correct = correct + 1;
    end
end

accuracy = ( correct / 21025 ) * 100;

end



