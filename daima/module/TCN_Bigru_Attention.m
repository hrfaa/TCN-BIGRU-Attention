%% 本程序为单特征输入
%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc  
addpath(genpath(pwd))    %读
%% 导入数据
% data = readmatrix('../nasa4.xlsx') %三个训练，一个验证
% data = data(1:636,3);
[h1,l1]=data_process(data,12); %步长为12
res = [h1,l1];
num_samples = size(res,1);   %样本个数

% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.75;                              % 训练集占数据集比例
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


%% 优化算法优化前，构建优化前的TCN_BiGRU_Attention模型


outputSize = 1;  %数据输出y的维度  
numFilters = 64;
filterSize = 5;
dropoutFactor = 0.005;
numBlocks = 2;
tempLayers = sequenceInputLayer(numFeatures,"Name","sequence");
lgraph = layerGraph(tempLayers);

outputName = tempLayers.Name;

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
    FlipLayer("flip3")
    gruLayer(35,"Name","gru2")];
lgraph = addLayers(lgraph,tempLayers);


tempLayers = [
    concatenationLayer(1,2,"Name","concat")
    selfAttentionLayer(2,50,"Name","selfattention")   %单头注意力Attention机制，把1改为2,3,4……即为多头，后面的50是键值
    fullyConnectedLayer(outdim,"Name","fc")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);


lgraph = connectLayers(lgraph,outputName,"flatten");
lgraph = connectLayers(lgraph,"flatten","gru1");
lgraph = connectLayers(lgraph,"flatten","flip3");
lgraph = connectLayers(lgraph,"gru1","concat/in1");
lgraph = connectLayers(lgraph,"gru2","concat/in2");


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

%  训练
tic
net = trainNetwork(vp_train, vt_train, lgraph, options0);
toc
% analyzeNetwork(net);% 查看网络结构
%  预测
t_sim1 = predict(net, vp_train); 
t_sim2 = predict(net, vp_test); 

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


% 指标计算
% disp('…………训练集误差指标…………')
% [mae1,rmse1,mape1,error1]=calc_error(T_train1,T_sim1);
% fprintf('\n')
% 
% figure('Position',[200,300,600,200])
% plot(T_train1);
% hold on
% plot(T_sim1)
% legend('真实值','预测值')
% title('TCN-BiGRU-ATTENTION训练集预测效果对比')
% xlabel('样本点')
% ylabel('容量')
% 
% disp('…………测试集误差指标…………')
% [mae2,rmse2,mape2,error2]=calc_error(T_test2,T_sim2);
% fprintf('\n')
% 
% figure;
% plotregression(T_test2,T_sim2,'回归图');
% set(gcf,'color','w')
% 
% 
% figure('Position',[200,300,600,200])
% plot(T_test2);
% hold on
% plot(T_sim2)
% legend('真实值','预测值')
% title('TCN-BiGRU-ATTENTION预测集预测效果对比')
% xlabel('样本点')
% ylabel('容量')
% 
% figure('Position',[200,300,600,200])
% plot(T_sim2-T_test2)
% title('TCN-BiGRU-ATTENTION误差曲线图')
% xlabel('样本点')
% ylabel('容量')

%% 优化TCN-BiGRU-Attention

disp(' ')
disp('优化TCN_BiLSTM_attention神经网络：')

%% 初始化参数 
popsize=6;   %初始种群规模 
maxgen=4;   %最大进化代数
fobj = @(x)objectiveFunction(x,numFeatures,outdim,vp_train,vt_train,vp_test,T_test,ps_output);
% 优化参数设置
lb = [0.0001 5 1  1]; %参数的下限。分别是学习率，biGRU的神经元个数，注意力机制的键值, 卷积核大小
ub = [0.01 50 50 10];    %参数的上限
dim = length(lb);%数量

[Best_score,Best_pos,curve]=SSA(popsize,maxgen,lb,ub,dim,fobj); %修改这里的函数名字即可
setdemorandstream(pi);

%% 把最佳参数Best_pos回带 
[~,optimize_T_sim] = objectiveFunction(Best_pos,numFeatures,outdim,vp_train,vt_train,vp_test,T_test,ps_output);
setdemorandstream(pi);

% str={'真实值','TCN-BiGRU-Attention','优化后TCN-BiGRU-Attention'};
% figure('Units', 'pixels', ...
%     'Position', [300 300 860 370]);
% plot(T_test,'-','Color',[0.8500 0.3250 0.0980]) 
% hold on
% plot(T_sim2,'-.','Color',[0.4940 0.1840 0.5560]) 
% hold on
% plot(optimize_T_sim,'-','Color',[0.4660 0.6740 0.1880])
% legend(str)
% set (gca,"FontSize",12,'LineWidth',1.2)
% box off
% legend Box off

% %% 回归图
% figure;
% plotregression(T_test2,optimize_T_sim,'回归图');  % optimimize为处理后预测数据  
% set(gcf,'color','w')

%%  第二优化算法
%% 初始化参数 
popsize=6;   %初始种群规模 
maxgen=6;   %最大进化代数
fobj = @(x)objectiveFunction(x,numFeatures,outdim,vp_train,vt_train,vp_test,T_test,ps_output);
% 优化参数设置
lb = [0.0001 1 1  1]; %参数的下限。分别是学习率，biGRU的神经元个数，注意力机制的键值, 卷积核大小
ub = [0.01 50 50 10];    %参数的上限
dim = length(lb);%数量

[Best_score2,Best_pos2,curve2]=SSABO(popsize,maxgen,lb,ub,dim,fobj); %修改这里的函数名字即可
setdemorandstream(pi);

[~,optimize_T_sim2] = objectiveFunction(Best_pos2,numFeatures,outdim,vp_train,vt_train,vp_test,T_test,ps_output);
setdemorandstream(pi);


%% 比较算法误差
test_y = T_test;
Test_all = [];

y_test_predict = T_sim2;  %sim2为原始模型预测结果
[test_MAE,test_MAPE,test_MSE,test_RMSE,test_R2]=calc_error(y_test_predict,test_y);
Test_all=[Test_all;test_MAE test_MAPE test_MSE test_RMSE test_R2];


y_test_predict = optimize_T_sim;
[test_MAE,test_MAPE,test_MSE,test_RMSE,test_R2]=calc_error(y_test_predict,test_y);
Test_all=[Test_all;test_MAE test_MAPE test_MSE test_RMSE test_R2];


y_test_predict = optimize_T_sim2;
[test_MAE,test_MAPE,test_MSE,test_RMSE,test_R2]=calc_error(y_test_predict,test_y);
Test_all=[Test_all;test_MAE test_MAPE test_MSE test_RMSE test_R2];
 	

str={'真实值','TCN-BiGRU-Attention','优化后TCN-BiGRU-Attention','最终结果'};
str1=str(2:end);
str2={'MAE','MAPE','MSE','RMSE','R2'};
data_out=array2table(Test_all);
data_out.Properties.VariableNames=str2;
data_out.Properties.RowNames=str1;
disp(data_out)

%% 柱状图 MAE MAPE RMSE 柱状图适合量纲差别不大的
color=    [0.66669    0.1206    0.108
    0.1339    0.7882    0.8588
    0.1525    0.6645    0.1290
    0.8549    0.9373    0.8275   
    0.1551    0.2176    0.8627
    0.7843    0.1412    0.1373
    0.2000    0.9213    0.8176
      0.5569    0.8118    0.7882
       1.0000    0.5333    0.5176];
figure('Units', 'pixels', ...
    'Position', [300 300 660 375]);
plot_data_t=Test_all(:,[1,2,4])';
b=bar(plot_data_t,0.8);
hold on

for i = 1 : size(plot_data_t,2)
    x_data(:, i) = b(i).XEndPoints'; 
end

for i =1:size(plot_data_t,2)
    b(i).FaceColor = color(i,:);
    b(i).EdgeColor=[0.3353    0.3314    0.6431];
    b(i).LineWidth=1.2;
end

for i = 1 : size(plot_data_t,1)-1
    xilnk=(x_data(i, end)+ x_data(i+1, 1))/2;
    b1=xline(xilnk,'--','LineWidth',1.2);
    hold on
end 

ax=gca;
legend(b,str1,'Location','best')
ax.XTickLabels ={'MAE', 'MAPE', 'RMSE'};
set(gca,"FontSize",10,"LineWidth",1)
box off
legend box off



% 二维图
figure
plot_data_t1=Test_all(:,[1,5])';
MarkerType={'s','o','pentagram','^','v'};
for i = 1 : size(plot_data_t1,2)
   scatter(plot_data_t1(1,i),plot_data_t1(2,i),120,MarkerType{i},"filled")
   hold on
end
set(gca,"FontSize",12,"LineWidth",2)
box off
legend box off
legend(str1,'Location','best')
xlabel('MAE')
ylabel('R2')
grid on




%% 雷达图
figure('Units', 'pixels', ...
    'Position', [150 150 520 500]);
Test_all1=Test_all./sum(Test_all);  %把各个指标归一化到一个量纲
Test_all1(:,end)=1-Test_all(:,end);
RC=radarChart(Test_all1);
str3={'MAE','MAPE','MSE','RMSE','R2'};
RC.PropName=str3;
RC.ClassName=str1;
RC=RC.draw(); 
RC.legend();
RC.setBkg('FaceColor',[1,1,1])
RC.setRLabel('Color','none')
colorList=[78 101 155;
          181 86 29;
          184 168 207;
          231 188 198;
          253 207 158;
          239 164 132;
          182 118 108]./255;

for n=1:RC.ClassNum
    RC.setPatchN(n,'Color',colorList(n,:),'MarkerFaceColor',colorList(n,:))
end







