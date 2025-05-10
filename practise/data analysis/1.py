import math
import random

# 随机种子使结果可复现
random.seed(0)

# 问题参数设置
T = 10
N = 3
V = 100               # 仓储容量上限
S = 100               # 每次开机的固定设置成本
H = 1.5               # 单位库存持有成本 (元/单位*期)

# 随机生成三个客户在10期的需求矩阵 D[n][t]
D = [[random.randint(5, 10) for t in range(T)] for n in range(N)]
# 计算每期总需求（3个客户需求之和）
total_demand = [sum(D[n][t] for n in range(N)) for t in range(T)]
# 设置每期产能上限 P_t，确保不低于该期需求（这里取稍高于需求的值）
P = [max(d, math.ceil(sum(total_demand)/T)) for d in total_demand]

# 随机生成运输成本矩阵 C[n][t]
C = [[random.uniform(1, 5) for t in range(T)] for n in range(N)]

# 评价函数：给定生产计划 x 序列，计算总成本（含惩罚）
def compute_cost(x):
    total_cost = 0.0
    I_prev = 0  # 上期末库存（初始库存）
    # 遍历每个时间周期，累加成本
    for t in range(T):
        # 产能约束惩罚: 若生产量超过产能P_t，则加大惩罚
        if x[t] > P[t]:
            return float('inf')  # 直接判为不可行（惩罚为无限大成本）
        # 生产成本
        total_cost += 1.0 * x[t]
        # 设置成本：如果本期生产量>0则计固定成本
        if x[t] > 0:
            total_cost += S
        # 库存更新：本期末库存 = 上期库存 + 本期生产 - 本期总需求
        I_t = I_prev + x[t] - total_demand[t]
        # 若出现库存负值，说明未满足当期需求（不可行）
        if I_t < 0:
            return float('inf')  # 不可行解的成本设为无穷大（惩罚）
        # 库存容量惩罚：超出仓储上限判为不可行
        if I_t > V:
            return float('inf')
        # 库存持有成本（按期末库存计）
        total_cost += H * I_t
        # 运输成本（按当期交付的需求计）
        # 需求均在当期交付，运输成本为 sum_n (D[n,t] * C[n,t])
        for n in range(N):
            total_cost += C[n][t] * D[n][t]
        # 准备下一期
        I_prev = I_t
    return total_cost

# 随机生成初始解：按各期需求量生产（满足需求但可能不是最优）
current_x = total_demand[:]  # 拷贝一份作为当前解
# 确保初始解不违反约束：若有库存或产能问题，这里简单处理或假设输入可行
current_cost = compute_cost(current_x)
if current_cost == float('inf'):
    # 如果按需求生产不可行，这里进行简单修正：如遇超产能需求则提前生产
    # （为了简化，逐期往前平移超出的产量直到可行）
    inventory = 0
    current_x = [0]*T
    for t in range(T):
        # 需求可能部分由库存满足
        needed = max(total_demand[t] - inventory, 0)
        # 尽可能生产需求（受产能限制）
        produce = min(needed, P[t])
        current_x[t] = produce
        # 更新库存，若生产不足会导致负库存（需求未满足），稍后用惩罚处理
        inventory = max(inventory + produce - total_demand[t], 0)
    current_cost = compute_cost(current_x)

# 模拟退火参数
T_init = 100.0    # 初始温度
T_min = 1e-3      # 终止温度
cooling_rate = 0.95
max_iter = 1000

best_x = current_x[:]        # 全局最优解（初始化为当前解）
best_cost = current_cost

T_current = T_init
iteration = 0

# 主循环：在温度降到阈值或达到最大迭代次数前持续迭代
while T_current > T_min and iteration < max_iter:
    # 在当前解邻域中随机产生一个新解
    new_x = current_x[:]
    # 随机选择一个扰动操作，这里采用在两个随机时期之间移动部分产量
    i = random.randrange(T)
    j = random.randrange(T)
    if i == j:
        continue  # 相同时期跳过一次
    if i > j:
        i, j = j, i  # 确保 i < j （将部分产量从后期提前到前期）
    # 计算可转移的最大产量 Δ：受限于 j 期原有产量和 i 期产能余量
    # 允许的最大转移量不能超过 j期原计划产量（不能为负）且不能使i期超产能
    max_shift = min(new_x[j], P[i] - new_x[i])
    if max_shift <= 0:
        # 无法转移产量则跳过本次扰动
        iteration += 1
        T_current *= cooling_rate
        continue
    # 随机选取一个 1 到 max_shift 之间的转移量
    delta = random.randint(1, int(max_shift))
    new_x[i] += delta   # 提前 delta 单位到时期 i 生产
    new_x[j] -= delta   # 减少时期 j 相应的生产量
    # 计算新解的成本
    new_cost = compute_cost(new_x)
    # 决定是否接受新解
    if new_cost < current_cost:
        # 更优解，直接接受
        current_x = new_x
        current_cost = new_cost
    else:
        # 更差解，以一定概率接受
        # 计算接受概率（Metropolis准则）
        delta_cost = new_cost - current_cost
        accept_prob = math.exp(-delta_cost / T_current) if new_cost != float('inf') else 0
        if random.random() < accept_prob:
            current_x = new_x
            current_cost = new_cost
    # 更新全局最佳解
    if current_cost < best_cost:
        best_cost = current_cost
        best_x = current_x[:]
    # 降温并迭代计数
    T_current *= cooling_rate
    iteration += 1

# 输出最优结果
print("最优生产批量 x_t 序列:", best_x)
print("对应总成本:", round(best_cost, 2))
