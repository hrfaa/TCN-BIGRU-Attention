
%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc  
addpath(genpath(pwd))

%% 导入数据
data = readmatrix('../data4.xlsx')
data = data(1:400,3);
% data = readmatrix('../siall.xlsx')
% data = data(1:300,3);
[h1,l1]=data_process(data,12); %步长为12
res = [h1,l1];
num_samples = size(res,1);   %样本个数

% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.7;                              % 训练集占数据集比例
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  格式转换
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end
numFeatures = size(p_train,1);

%% 优化算法优化前，构建优化前的TCN_GRU-ATTENTION模型

outputSize = 1;  %数据输出y的维度  
numFilters = 64;
filterSize = 5;
dropoutFactor = 0.005;
numBlocks = 2;

layer = sequenceInputLayer(numFeatures,Normalization="rescale-symmetric",Name="input");
lgraph = layerGraph(layer);

outputName = layer.Name;

for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    
    layers = [
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
        layerNormalizationLayer
        dropoutLayer(dropoutFactor) 
        % spatialDropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
        layerNormalizationLayer
        reluLayer
        dropoutLayer(dropoutFactor) 
        additionLayer(2,Name="add_"+i)];

    % Add and connect layers.
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);

    % Skip connection.
    if i == 1
        % Include convolution in first skip connection.
        layer = convolution1dLayer(1,numFilters,Name="convSkip");

        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,outputName,"convSkip");
        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
    else
        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
    end
    
    % Update layer output name.
    outputName = "add_" + i;
end

tempLayers = flattenLayer("Name","flatten");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = gruLayer(35,"Name","gru1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    selfAttentionLayer(1,50,"Name","selfattention")   % %单头注意力Attention机制，把1改为2,3,4……即为多头，后面的50是键值
    fullyConnectedLayer(outdim,"Name","fc")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);


lgraph = connectLayers(lgraph,outputName,"flatten");
lgraph = connectLayers(lgraph,"flatten","gru1");
lgraph = connectLayers(lgraph,"gru1","selfattention");



%  参数设置
options0 = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 150, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.01, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod',100, ...                   % 训练100次后开始调整学习率
    'LearnRateDropFactor',0.001, ...                    % 学习率调整因子
    'L2Regularization', 0.001, ...         % 正则化参数
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 1, ...                                 % 关闭优化过程
    'Plots', 'none');                    % 画出曲线

% 网络训练
tic
net0 = trainNetwork(vp_train,vt_train,lgraph,options0);
toc
% analyzeNetwork(net0);% 查看网络结构
%  预测
t_sim1 = predict(net0, vp_train); 
t_sim2 = predict(net0, vp_test); 

%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_train1 = T_train;
T_test2 = T_test;

%  数据格式转换
T_sim1 = cell2mat(T_sim1);% cell2mat将cell元胞数组转换为普通数组
T_sim2 = cell2mat(T_sim2);

T_sim1 = T_sim1';
T_sim2 = T_sim2';

TCN_BiGRU_ATTENTION_TSIM1 = T_sim1;
TCN_BiGRU_ATTENTION_TSIM2 = T_sim2;
save TCN_BiGRU_ATTENTION TCN_BiGRU_ATTENTION_TSIM1 TCN_BiGRU_ATTENTION_TSIM2




%% 初始化参数 
popsize=6;   %初始种群规模 
maxgen=4;   %最大进化代数
fobj = @(x)objectiveFunction(x,numFeatures,outdim,vp_train,vt_train,vp_test,T_test,ps_output);
% 优化参数设置
lb = [0.001 10 2  2]; %参数的下限。分别是学习率，biGRU的神经元个数，注意力机制的键值, 卷积核大小
ub = [0.01 50 50 10];    %参数的上限s
dim = length(lb);%数量

[Best_score2,Best_pos2,curve2]=SSABO(popsize,maxgen,lb,ub,dim,fobj); %修改这里的函数名字即可
setdemorandstream(pi);

[~,optimize_T_sim2] = objectiveFunction(Best_pos2,numFeatures,outdim,vp_train,vt_train,vp_test,T_test,ps_output);
setdemorandstream(pi);

figure
plot(curve2,'r-','linewidth',2)
xlabel('进化代数')
ylabel('均方误差')
legend('最佳适应度')
title('进化曲线')

disp('最终优化的超参数值为：');
disp(Best_pos2);

%% 比较算法误差
test_y = T_test;
Test_all = [];

y_test_predict = T_sim2;  %sim2为原始模型预测结果
[test_MAE,test_MAPE,test_MSE,test_RMSE,test_R2]=calc_error(y_test_predict,test_y);
Test_all=[Test_all;test_MAE test_MAPE test_MSE test_RMSE test_R2];

y_test_predict = optimize_T_sim2;
[test_MAE,test_MAPE,test_MSE,test_RMSE,test_R2]=calc_error(y_test_predict,test_y);
Test_all=[Test_all;test_MAE test_MAPE test_MSE test_RMSE test_R2];
 	

str={'真实值','BiGRU-ATTENTION','最终结果'};
str1=str(2:end);
str2={'MAE','MAPE','MSE','RMSE','R2'};
data_out=array2table(Test_all);
data_out.Properties.VariableNames=str2;
data_out.Properties.RowNames=str1;
disp(data_out)