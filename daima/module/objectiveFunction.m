% 

function [MAPE,T_sim]= objectiveFunction(x,numFeatures,outdim,vp_train,vt_train,vp_test,T_test,ps_output)

%% 将优化目标参数传进来的值 转换为需要的超参数
learning_rate = x(1);            %% 学习率
NumNeurons = round(x(2));        %% biGRU神经元个数
keys = round(x(3));        %% 自注意力机制的键值数
ksize = round(x(4));   %卷积核大小
setdemorandstream(pi);

%% 优化算法优化前，构建优化前的TCN_BiGRU_Attention模型


outputSize = 1;  %数据输出y的维度  
numFilters = 64;
filterSize = 5;
dropoutFactor = 0.005;
numBlocks = 2;
% lgraph = layerGraph();
% tempLayers = sequenceInputLayer(f_,Normalization="rescale-symmetric",Name="input");
tempLayers = sequenceInputLayer(numFeatures,"Name","sequence");
lgraph = layerGraph(tempLayers);

outputName = tempLayers.Name;

for i = 1:numBlocks
    dilationFactor = 2^(i-1);
    
    layers = [
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
        layerNormalizationLayer
        dropoutLayer(dropoutFactor) 
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
    selfAttentionLayer(2,50,"Name","selfattention")   
    fullyConnectedLayer(outdim,"Name","fc")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% % 清理辅助变量
clear tempLayers;
lgraph = connectLayers(lgraph,outputName,"flatten");
lgraph = connectLayers(lgraph,"flatten","gru1");
lgraph = connectLayers(lgraph,"flatten","flip3");
lgraph = connectLayers(lgraph,"gru1","concat/in1");
lgraph = connectLayers(lgraph,"gru2","concat/in2");


%%  参数设置
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 30, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', learning_rate, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod', 22, ...                   % 训练60次后开始调整学习率
    'LearnRateDropFactor',0.1, ...                    % 学习率调整因子
    'ExecutionEnvironment', 'cpu',...                 % 训练环境
    'Verbose', 1, ...                                 % 关闭优化过程
    'Plots', 'none');                    % 画出曲线

%  训练

net = trainNetwork(vp_train,vt_train,lgraph, options);

% analyzeNetwork(net);% 查看网络结构 
                     
%% 测试与评估
t_sim = net.predict(vp_test);  

%  数据反归一化
T_sim = mapminmax('reverse', t_sim, ps_output);

T_sim = cell2mat(T_sim);% cell2mat将cell元胞数组转换为普通数组
T_sim = T_sim';
%% 计算误差
MAPE=sum(abs((T_sim-T_test)./T_test))/length(T_test);  % 平均百分比误差
display(['本批次MAPE:', num2str(MAPE)]);
end
