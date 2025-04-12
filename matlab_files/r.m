function R = calculate_reflectance(lambda, theta, d1, d2, d3, n1, n2, n3, n0, ns)
    % calculate_reflectance - 计算三层膜系统的反射率（TE 极化）
    %
    % 输入:
    %   lambda - 波长 (m)
    %   theta  - 入射角 (rad)
    %   d1, d2, d3 - 每层膜的厚度 (m)
    %   n1, n2, n3 - 每层膜的折射率
    %   n0 - 前侧介质（空气）折射率
    %   ns - 后侧介质（基底）折射率
    %
    % 输出:
    %   R - 反射率

    % 波数
    k = 2 * pi / lambda;

    % 计算每层膜内的折射角（处理全反射）
    theta1 = calculate_transmission_angle(n0, theta, n1);
    theta2 = calculate_transmission_angle(n0, theta, n2);
    theta3 = calculate_transmission_angle(n0, theta, n3);
    thetas = calculate_transmission_angle(n0, theta, ns);

    % 界面反射系数（TE 极化）
    r01 = calculate_reflection_coefficient(n0, theta, n1, theta1);
    r12 = calculate_reflection_coefficient(n1, theta1, n2, theta2);
    r23 = calculate_reflection_coefficient(n2, theta2, n3, theta3);
    r3s = calculate_reflection_coefficient(n3, theta3, ns, thetas);

    % 透射系数（TE 极化）
    t01 = calculate_transmission_coefficient(n0, theta, n1, theta1);
    t12 = calculate_transmission_coefficient(n1, theta1, n2, theta2);
    t23 = calculate_transmission_coefficient(n2, theta2, n3, theta3);
    t3s = calculate_transmission_coefficient(n3, theta3, ns, thetas);

    % 界面矩阵 D_ij
    D01 = [1/t01, r01/t01; r01/t01, 1/t01];
    D12 = [1/t12, r12/t12; r12/t12, 1/t12];
    D23 = [1/t23, r23/t23; r23/t23, 1/t23];
    D3s = [1/t3s, r3s/t3s; r3s/t3s, 1/t3s];

    % 传播矩阵 P_j（含复数相位延迟）
    phi1 = k * n1 * d1 * calculate_cos_theta(theta1);
    phi2 = k * n2 * d2 * calculate_cos_theta(theta2);
    phi3 = k * n3 * d3 * calculate_cos_theta(theta3);
    P1 = [exp(-1i * phi1), 0; 0, exp(1i * phi1)];
    P2 = [exp(-1i * phi2), 0; 0, exp(1i * phi2)];
    P3 = [exp(-1i * phi3), 0; 0, exp(1i * phi3)];

    % 总传递矩阵 S（严格顺序：入射到基底）
    S = D01 * P1 * D12 * P2 * D23 * P3 * D3s;

    % 反射系数 r 和反射率 R
    r = S(2,1) / S(1,1);
    R = abs(r)^2;
end

% 辅助函数：计算折射角（全反射时返回复数角度）
function theta_j = calculate_transmission_angle(n_i, theta_i, n_j)
    sin_theta_j = n_i * sin(theta_i) / n_j;
    if sin_theta_j > 1
        theta_j = asin(1) + 1i * acosh(sin_theta_j); % 复数角度
    else
        theta_j = asin(sin_theta_j);
    end
end

% 辅助函数：计算 cos(theta_j)（处理复数）
function cos_theta_j = calculate_cos_theta(theta_j)
    if isreal(theta_j)
        cos_theta_j = cos(theta_j);
    else
        cos_theta_j = 1i * sqrt(sin(theta_j)^2 - 1); % 全反射时 cosθ 为纯虚数
    end
end

% 辅助函数：计算界面反射系数（TE 极化）
function r_ij = calculate_reflection_coefficient(n_i, theta_i, n_j, theta_j)
    cos_theta_i = calculate_cos_theta(theta_i);
    cos_theta_j = calculate_cos_theta(theta_j);
    r_ij = (n_i * cos_theta_i - n_j * cos_theta_j) / (n_i * cos_theta_i + n_j * cos_theta_j);
end

% 辅助函数：计算界面透射系数（TE 极化）
function t_ij = calculate_transmission_coefficient(n_i, theta_i, n_j, theta_j)
    cos_theta_i = calculate_cos_theta(theta_i);
    cos_theta_j = calculate_cos_theta(theta_j);
    t_ij = 2 * n_i * cos_theta_i / (n_i * cos_theta_i + n_j * cos_theta_j);
end

% 设置随机种子以确保可重现性
rng(42);

% 参数设置 - 固定入射角为0度(法向入射)
% 只生成不同折射率和厚度的组合

% 折射率范围: 从1到2，采用16个等间隔点
n_values = linspace(1, 2, 16);  % 生成16个均匀分布的折射率值

% 厚度范围: 从20nm到200nm，采用16个等间隔点
d_values = linspace(20, 200, 16);  % 生成16个均匀分布的厚度值

% 固定入射角为0度(法向入射)
fixed_theta_deg = 0;  % 固定的入射角

% 生成参数网格 - 这将创建以下数量的样本:
% 16^3 * 16^3 = 4096 * 4096 = 16,777,216个样本
[n1_grid, n2_grid, n3_grid, d1_grid, d2_grid, d3_grid] = ndgrid(...
    n_values, n_values, n_values, d_values, d_values, d_values);

% 转换为列向量以便遍历 - 只有6个参数(三层膜的折射率和厚度)
full_params = [n1_grid(:), n2_grid(:), n3_grid(:), ...
          d1_grid(:), d2_grid(:), d3_grid(:)];

% 随机挑选约50,000个样本
num_samples = 50000;
total_generated = size(full_params, 1);
if total_generated > num_samples
    % 随机挑选指定数量的样本
    sample_indices = randperm(total_generated, num_samples);
    params = full_params(sample_indices, :);
    
    fprintf('\n\n从总共 %d 个样本中随机挑选了 %d 个样本\n\n', ...
        total_generated, num_samples);
else
    params = full_params;
    fprintf('\n\n生成了全部 %d 个样本\n\n', total_generated);
end

% 预分配结果数组
num_rows = size(params, 1);
results = zeros(num_rows, 7);  % 7列: n1, n2, n3, d1, d2, d3, R

fprintf('\n开始计算 %d 个数据点...\n', num_rows);

% 使用进度条来显示计算进度
progress_interval = max(1, floor(num_rows/100));

% 并行计算（需要Parallel Computing Toolbox）
parfor i = 1:num_rows
    % 提取参数 - 固定入射角为0度
    n1 = params(i, 1);
    n2 = params(i, 2);
    n3 = params(i, 3);
    d1 = params(i, 4) * 1e-9;  % nm转m
    d2 = params(i, 5) * 1e-9;
    d3 = params(i, 6) * 1e-9;
    theta = deg2rad(fixed_theta_deg);  % 固定入射角为0度
    
    % 计算反射率
    lambda = 550e-9;  % 固定波长
    n0 = 1.0;         % 空气折射率
    ns = 1.5;         % 基底折射率
    
    try
        R = calculate_reflectance(lambda, theta, d1, d2, d3, n1, n2, n3, n0, ns);
    catch
        R = NaN;       % 处理可能的数值错误
    end
    
    % 存储结果 - 去除入射角，只保疖6个输入和1个输出
    results(i, :) = [n1, n2, n3, params(i, 4), params(i, 5), params(i, 6), R];
    
    % 每计算一定数量的样本显示进度
    if mod(i, progress_interval) == 0
        fprintf('\r计算进度: %.1f%% (%d/%d)', 100*i/num_rows, i, num_rows);
    end
end

% 清除最后的进度条并显示统计信息
fprintf('\n\n计算完成! 生成了 %d 个数据点\n', num_rows);

% 显示反射率的基本统计信息
valid_reflectance = results(:, 7);
valid_reflectance = valid_reflectance(~isnan(valid_reflectance));
fprintf('\n反射率统计信息:\n');
fprintf('  最小值: %.6f\n', min(valid_reflectance));
fprintf('  最大值: %.6f\n', max(valid_reflectance));
fprintf('  平均值: %.6f\n', mean(valid_reflectance));
fprintf('  标准差: %.6f\n\n', std(valid_reflectance));

% 添加列名以便于处理
column_headers = {'n1', 'n2', 'n3', 'd1_nm', 'd2_nm', 'd3_nm', 'reflectance'};

% 把结果转换为表格，并添加列名
resultsTable = array2table(results, 'VariableNames', column_headers);

% 保存为CSV文件（文件名带时间戳以防覆盖）
filename = sprintf('reflectance_data_fixed0deg_%s_n%d.csv', datestr(now, 'yyyymmdd_HHMMSS'), num_rows);
writetable(resultsTable, filename);

% 显示完成提示
disp(['计算完成，结果已保存至: ', filename]);