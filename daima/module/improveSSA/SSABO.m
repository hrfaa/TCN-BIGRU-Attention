function [fMin, bestX, Convergence_curve] = SSA_GPR(pop, M, c, d, dim, fobj)

%% 参数初始化
P_percent = 0.2;    % 生产者数量占总种群的比例
pNum = round(pop * P_percent);    % 生产者数量

lb= c.*ones( 1,dim );    % Lower limit/bounds/     a vector   上下界
ub= d.*ones( 1,dim );    % Upper limit/bounds/     a vector

% 初始化种群位置和适应度
x = lb + (ub - lb) .* rand(pop, dim);    % 种群位置
fit = arrayfun(@(i) fobj(x(i, :)), 1:pop);    % 种群适应度

pFit = fit;
pX = x;    % 个体的最优位置

% 找到全局最优解
[fMin, bestI] = min(fit);
bestX = x(bestI, :);    % 全局最优位置

% 收敛曲线初始化
Convergence_curve = zeros(M, 1);

%% 迭代优化
for t = 1 : M
    % 对每个个体进行排序
    [ans, sortIndex] = sort(pFit);
    
    % 找到最差解
    [fmax, B] = max(pFit);
    worse = x(B, :);

    % 更新位置
    for i = 1 : pNum
        r2 = rand(1);
        if r2 < 0.8
            % 使用指数衰减因子更新位置
            r1 = rand(1);
            x(sortIndex(i), :) = pX(sortIndex(i), :) .* exp(-(i) / (r1 * M));
        else
            % 随机扰动更新位置
            x(sortIndex(i), :) = pX(sortIndex(i), :) + randn(1, dim);
        end
        % 应用边界限制
        x(sortIndex(i), :) = Bounds(x(sortIndex(i), :), lb, ub);
        % 计算新的适应度
        fit(sortIndex(i)) = fobj(x(sortIndex(i), :));
    end

    % 更新全局最优解
    [fMMin, bestII] = min(fit);
    bestXX = x(bestII, :);

    for i = (pNum + 1) : pop
    A = floor(rand(1, dim) * 2) * 2 - 1; % 确保A是1x4向量
    if i > (pop / 2)
        % 使用高斯扰动更新位置
        x(sortIndex(i), :) = randn(1, dim) .* exp((worse - pX(sortIndex(i), :)) / (i)^2);
    else
        % 使用线性组合更新位置
        % 确保linearCombination是一个4x1向量
        linearCombination = A / norm(A); % 归一化A
        x(sortIndex(i), :) = bestXX + (abs(pX(sortIndex(i), :) - bestXX)) .* linearCombination;
    end
    % 应用边界限制
    x(sortIndex(i), :) = Bounds(x(sortIndex(i), :), lb, ub);
    % 计算新的适应度
    fit(sortIndex(i)) = fobj(x(sortIndex(i), :));
end

    % 选择一部分个体进行局部搜索
    c = randperm(numel(sortIndex));
    b = sortIndex(c(1:round(pop * 0.3)));
    for j = 1 : length(b)
        if pFit(sortIndex(b(j))) > fMin
            % 使用全局最优解进行引导
            x(sortIndex(b(j)), :) = bestX + (randn(1, dim) .* abs(pX(sortIndex(b(j)), :) - bestX));
        else
            % 使用最差解进行引导
            x(sortIndex(b(j)), :) = pX(sortIndex(b(j)), :) + (2 * rand(1) - 1) .* (abs(pX(sortIndex(b(j)), :) - worse)) / (pFit(sortIndex(b(j))) - fmax + 1e-50);
        end
        % 应用边界限制
        x(sortIndex(b(j)), :) = Bounds(x(sortIndex(b(j)), :), lb, ub);
        % 计算新的适应度
        fit(sortIndex(b(j))) = fobj(x(sortIndex(b(j)), :));
    end

    % 更新个体的最优解
    for i = 1 : pop
        if fit(i) < pFit(i)
            pFit(i) = fit(i);
            pX(i, :) = x(i, :);
        end
        if pFit(i) < fMin
            fMin = pFit(i);
            bestX = pX(i, :);
        end
    end

    % 记录收敛曲线
    Convergence_curve(t) = fMin;
    disp([num2str(t), '次迭代的RMSE为：', num2str(fMin)]);
end

%% 应用边界限制函数
function s = Bounds(s, Lb, Ub)
    % 应用下界限制
    temp = s;
    I = temp < Lb;
    temp(I) = Lb(I);

    % 应用上界限制
    J = temp > Ub;
    temp(J) = Ub(J);

    % 更新位置
    s = temp;

