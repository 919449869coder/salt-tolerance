%% 加载数据
% 训练集图像是160异常+160正常, 测试集图像是40异常+40正常
% train_data里, 是160列异常+160列正常, test_data里, 是40异常+40正常 图像和序列一一对应
%% 加载训练集
clear; clc;
% load('train_data.mat'); % 一列是一个样本，每个样本500个数据点，320本个样
%      
load('train_datadata.mat');
train_data=train_datadata(2:end,8:end);
P_train=train_data';
M = size(P_train, 2);
% traindata = arrayDatastore(train_data); % 训练集序列：得先弄成一行是一个样本，再转化成arrayDatastore
image_traindata=load('heimgalltrain.mat');
image_traindata=struct2cell(image_traindata);
image_traindata=image_traindata{1};
image_traindata=image_traindata';

% imgsTrain = imageDatastore('训练集时频图','IncludeSubfolders' ,true);% 训练集图像

%imgsTrain = imageDatastore('F:\耐盐\耐盐\光谱\SPECTRALSHUJU\图像','IncludeSubfolders' ,true);% 训练集图像如果是jpg png图像读成这个格式 但是我的是存储好的761个cell

img=image_traindata(:,1);
num = size(img,1);
for i = 1:num    

    img{i} = imresize(img{i}, [100,100],'nearest');
end
% IMGtrainlgui=img(labell(30:end),1);
% IMGvalgui=img(labell(1:29),1);


 
% % 设置保存.mat文件的路径
% savePath = 'SPECTRALSHUJU\图像\';
% 
% % 循环遍历cell数组，并按顺序保存
% for k = 1:length(img)
%     % 创建文件名
%     fileName = sprintf('variable%d.mat', k);
%     % 完整的文件路径
%     fullFileName = fullfile(savePath, fileName);
%     % 保存当前cell中的变量
%     imgnew=img{k};
%     save(fullFileName, 'imgnew');
% end


%% 数据扩增
for i=1:length(img)
guizhuan90{i}=imrotate(img{i},90);
guizhuan180{i}=imrotate(img{i},180);
guizhuan270{i}=imrotate(img{i},270);
end
% 
KUOIMG=[img;guizhuan90'];
KUOIMG=[KUOIMG;guizhuan180'];
KUOIMG=[KUOIMG;guizhuan270'];

num1= size(img,1);
clear x
for i = 1:num1
    XTrain(:,:,:,i) = img{i};
end
%%%%转化成四维图像了
% imgsTrain = imageDatastore(x);% 训练集图像
% imageDatastore(img)
% =x;
% yy=[IMGlabeltrain;IMGlabeltrain];
% yy=[yy;IMGlabeltrain];
% yy=[yy;IMGlabeltrain];


%% 训练集标签
train_label = categorical(train_datadata(2:end,5)); % 训练集标签：0对应异常，1对应正常
trainlabel = arrayDatastore(train_label); % 得先是一列categorical，再转化
t_train =  categorical(train_label)';

% 合并训练集
Train = combine(XTrain,traindata,trainlabel);%% 图像序列合并??????/ 有问题

%% 加载测试集
load('test_datadata.mat');
test_data=test_datadata(2:end,8:end);
% testdata = arrayDatastore(test_data); % 测试集序列：得先弄成一行是一个样本，再转化 正常是序列是arrayDatastore，图像是imgsDatastore
P_test = test_data';
N = size(P_test , 2);
P_test  = mapminmax('apply', P_test, ps_input);

image_testdata=load('heimgalltest.mat');
image_testdata=struct2cell(image_testdata);
image_testdata=image_testdata{1};
image_testdata=image_testdata';
% idx = randperm(size(XTrain,4),100);

XVali =image_testdata(:,:,:,:);%%测试集
%for j = 1:length(XVali )  
%    XVali{j} = imresize(XVali{j}, [100,100],'nearest');
%end %%%缩小尺寸
%%  数据平铺
%   将数据平铺成1维数据只是一种处理方式
%   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
%   但是应该始终和输入层数据结构保持一致
num_dim=size(P_train, 1);               
p_train =  double(reshape(P_train, num_dim, 1, 1, M));
p_test  =  double(reshape(P_test , num_dim, 1, 1, N));

%% 验证集
clear XValidation
for k = 1:length(XVali)
    XValidation(:,:,:,k) = XVali{k};
end
% XTrain(:,:,:,idx) = [];
% YValidation = y(idx);
% y(idx) = [];

%     'RandXScale', [0.5 1], ...
%     'RandYScale', [0.5 1], ...
test_label = categorical(test_datadata(2:end,5)); % 训练集标签：0对应异常，1对应正常
testlabel = arrayDatastore(test_label); % 得先是一列categorical，再转化
t_test  =  categorical(test_label');

imageSize = [100 100 19];

imgsTest=XValidation;

test_label = categorical(test_datadata(2:end,5)); % 训练集标签：0对应异常，1对应正常
testlabel = arrayDatastore(test_label); % 得先是一列categorical，再转化
% 合并测试集
Test = combine(imgsTest,testdata,testlabel);%%%%问题  ？？？？？？？？？？？？

%% 构建双输入网络，一边是序列输入，一边是图像输入
lgraph = layerGraph();

% 添加层分支 将网络分支添加到层次图中
tempLayers = [
    imageInputLayer([100 100 19],"Name","图像输入") % 图像输入尺寸224*224*3
    convolution2dLayer([3 3],8,"Name","conv") % 3*3卷积，8个
    reluLayer("Name","relu")
    maxPooling2dLayer([11 11],"Name","maxpool","Stride",[11 11]) % 11*11池化，步长11
    convolution2dLayer([3 3],4,"Name","conv_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([3 3],"Name","maxpool_2","Stride",[2 2])
    flattenLayer("Name","flatten")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    imageInputLayer([1 39 1],"Name","序列输入") % 序列长度1*500*1，这里是把序列当成1*500*1的图像输入，这样跟一维卷积没有任何区别
    convolution2dLayer([1 7],8,"Name","conv_1") % 1*7卷积，8个
    reluLayer("Name","relu_1")
    maxPooling2dLayer([1 64],"Name","maxpool_1","Stride",[64 64]) % 1*64池化，步长64
    flattenLayer("Name","flatten_1")];
lgraph = addLayers(lgraph,tempLayers);


tempLayers = [
    concatenationLayer(1,2,"Name","concat") % 拼接层
    fullyConnectedLayer(2,"Name","fc") % 全连接，2代表2分类
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

% 连接层分支, 连接网络的所有分支以创建网络图。
lgraph = connectLayers(lgraph,"flatten_1","concat/in2");
lgraph = connectLayers(lgraph,"flatten","concat/in1");

% 显示网络结构
analyzeNetwork(lgraph);

% 构建优化器
options = trainingOptions('sgdm',... % 随机梯度下降法
    'MiniBatchSize',20,... % 批大小
    'MaxEpochs',15,... % 轮数
    'InitialLearnRate',0.001,... % 学习率
    'Verbose',1,...
    'ExecutionEnvironment','auto',... % 自动选CPU/GPU运行
    'Plots','training-progress');

%% 训练

trainedShuangNet = trainNetwork(Train,lgraph,options);%p_train, t_train 合并好的训练
%trainedShuangNet = trainNetwork([XTrain;p_train],train_label,lgraph,options);%p_train, t_train
%% 测试
y_pred = classify(trainedShuangNet,Test);
accuracy = mean(y_pred == test_label);
disp(['测试集准确率：',num2str(100*accuracy),'%']);
%% 混淆矩阵
plotconfusion(test_label, y_pred);



