import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 高斯分布函數
def gaussian(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

# 設定參數
x = np.linspace(-3, 3, 500)
mean = 0
std_initial = 0.5
std_final = 1.5

# 高斯分布數據
gaussian_initial = gaussian(x, mean, std_initial)
gaussian_final = gaussian(x, mean, std_final)

# 模擬 SDE 數據
t = np.linspace(0, 1, 100)
num_paths = 10
np.random.seed(42)
paths = np.zeros((len(t), num_paths))
paths[0, :] = np.random.normal(mean, std_initial, num_paths)

for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    paths[i, :] = paths[i - 1, :] + (-paths[i - 1, :] * dt) + np.sqrt(dt) * np.random.normal(0, 1, num_paths)

# 繪圖
fig, ax = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 3, 1]})

# 左側高斯分布
ax[0].plot(gaussian_initial, x, color="green", label="Initial Gaussian")
ax[0].fill_betweenx(x, 0, gaussian_initial, color="green", alpha=0.3)
ax[0].set_xlim(0, max(gaussian_initial) * 1.2)
ax[0].set_xlabel(r"$p_0(x)$")
ax[0].invert_xaxis()

# 右側高斯分布
ax[2].plot(gaussian_final, x, color="blue", label="Final Gaussian")
ax[2].fill_betweenx(x, 0, gaussian_final, color="blue", alpha=0.3)
ax[2].set_xlim(0, max(gaussian_final) * 1.2)
ax[2].set_xlabel(r"$p_T(x)$")

# 中間 SDE 路徑
norm = Normalize(vmin=t.min(), vmax=t.max())
colors = plt.cm.plasma(norm(t))
for path in paths.T:
    ax[1].plot(t, path, color="red", alpha=0.6)
ax[1].imshow(
    np.tile(colors, (100, 1)).T,
    extent=[t.min(), t.max(), x.min(), x.max()],
    aspect="auto",
    origin="lower",
    cmap="plasma",
    alpha=0.3,
)
ax[1].set_xlabel(r"$t$")
ax[1].set_ylabel(r"$x$")
ax[1].set_title("SDE Paths")

# 調整佈局
plt.tight_layout()
plt.savefig("sde_simulation_wgaussion.png", dpi=300)
plt.show()




# import numpy as np
# import matplotlib.pyplot as plt

# # 定義SDE的參數
# def forward_sde(x, t, dt, g):
#     """正向SDE"""
#     return x + (-0.5 * x * t) * dt + g * np.sqrt(dt) * np.random.randn(*x.shape)

# def reverse_sde(x, t, dt, g):
#     """反向SDE"""
#     return x + (0.5 * x * t) * dt - g * np.sqrt(dt) * np.random.randn(*x.shape)

# # 模擬參數
# n_samples = 10  # 軌跡數量
# n_steps = 100   # 時間步數
# T = 1.0         # 時間總長度
# g = 0.5         # 噪聲強度
# x_dim = 1       # 資料維度
# time = np.linspace(0, T, n_steps)
# dt = T / n_steps

# # 初始化資料分佈
# x0 = np.random.normal(size=(n_samples, x_dim))
# xt = x0.copy()

# # 儲存軌跡
# trajectories_forward = [xt]
# trajectories_reverse = []

# # 正向SDE模擬
# for t in range(n_steps):
#     xt = forward_sde(xt, time[t], dt, g)
#     trajectories_forward.append(xt)

# # 反向SDE模擬
# xt_reverse = trajectories_forward[-1].copy()
# trajectories_reverse.append(xt_reverse)
# for t in reversed(range(n_steps)):
#     xt_reverse = reverse_sde(xt_reverse, time[t], dt, g)
#     trajectories_reverse.append(xt_reverse)

# # 轉換為陣列
# trajectories_forward = np.array(trajectories_forward)
# trajectories_reverse = np.array(trajectories_reverse)

# # 畫圖
# plt.figure(figsize=(12, 6))
# colors = plt.cm.viridis(np.linspace(0, 1, n_steps))

# # 正向SDE圖
# plt.subplot(1, 2, 1)
# for i in range(n_samples):
#     plt.plot(trajectories_forward[:, i, 0], color="red", alpha=0.6, label="SDE" if i == 0 else None)
# plt.title("Forward SDE")
# plt.xlabel("Time")
# plt.ylabel("x")
# plt.grid()

# # 反向SDE圖
# plt.subplot(1, 2, 2)
# for i in range(n_samples):
#     plt.plot(trajectories_reverse[:, i, 0], color="blue", alpha=0.6, label="SDE" if i == 0 else None)
# plt.title("Reverse SDE")
# plt.xlabel("Time")
# plt.ylabel("x")
# plt.grid()

# plt.tight_layout()
# plt.savefig("sde_simulation.png", dpi=300)
# plt.show()


# 2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal

# # 定義SDE的參數
# def forward_sde(x, t, dt, g):
#     """正向SDE"""
#     return x + (-0.5 * x * t) * dt + g * np.sqrt(dt) * np.random.randn(*x.shape)

# def reverse_sde(x, t, dt, g):
#     """反向SDE"""
#     return x + (0.5 * x * t) * dt - g * np.sqrt(dt) * np.random.randn(*x.shape)

# # 模擬參數
# n_samples = 10  # 軌跡數量
# n_steps = 100   # 時間步數
# T = 1.0         # 時間總長度
# g = 0.5         # 噪聲強度
# x_dim = 1       # 資料維度
# time = np.linspace(0, T, n_steps)
# dt = T / n_steps

# # 初始化資料分佈
# x0 = np.random.normal(size=(n_samples, x_dim))
# xt = x0.copy()

# # 儲存軌跡
# trajectories_forward = [xt]
# trajectories_reverse = []

# # 正向SDE模擬
# for t in range(n_steps):
#     xt = forward_sde(xt, time[t], dt, g)
#     trajectories_forward.append(xt)

# # 反向SDE模擬
# xt_reverse = trajectories_forward[-1].copy()
# trajectories_reverse.append(xt_reverse)
# for t in reversed(range(n_steps)):
#     xt_reverse = reverse_sde(xt_reverse, time[t], dt, g)
#     trajectories_reverse.append(xt_reverse)

# # 轉換為陣列
# trajectories_forward = np.array(trajectories_forward)
# trajectories_reverse = np.array(trajectories_reverse)

# # 定義2D高斯分佈
# def gaussian_density(x, mean, cov):
#     """計算多變量高斯分佈密度"""
#     return multivariate_normal.pdf(x, mean=mean, cov=cov)

# # 建立2D背景
# x_range = np.linspace(-3, 3, 100)
# y_range = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(x_range, y_range)
# pos = np.dstack((X, Y))
# mean = [0, 0]  # 高斯分布的平均值
# cov = [[1, 0], [0, 1]]  # 共變異數矩陣
# Z = gaussian_density(pos, mean, cov)

# # 畫圖
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # 正向SDE圖
# ax = axs[0]
# ax.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.7)  # 背景熱圖
# for i in range(n_samples):
#     ax.plot(trajectories_forward[:, i, 0], np.zeros_like(trajectories_forward[:, i, 0]), color="red", alpha=0.6, label="SDE" if i == 0 else None)
# ax.set_title("Forward SDE")
# ax.set_xlabel("Time")
# ax.set_ylabel("x")
# ax.grid()

# # 反向SDE圖
# ax = axs[1]
# ax.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.7)  # 背景熱圖
# for i in range(n_samples):
#     ax.plot(trajectories_reverse[:, i, 0], np.zeros_like(trajectories_reverse[:, i, 0]), color="blue", alpha=0.6, label="SDE" if i == 0 else None)
# ax.set_title("Reverse SDE")
# ax.set_xlabel("Time")
# ax.set_ylabel("x")
# ax.grid()

# # 儲存圖片
# plt.tight_layout()
# plt.savefig("sde_with_gaussian_background.png", dpi=300)  # 儲存圖片
# plt.show()


# 3.
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal

# # 定義SDE的參數
# def forward_sde(x, t, dt, g):
#     """正向SDE"""
#     return x + (-0.5 * x * t) * dt + g * np.sqrt(dt) * np.random.randn(*x.shape)

# def reverse_sde(x, t, dt, g):
#     """反向SDE"""
#     return x + (0.5 * x * t) * dt - g * np.sqrt(dt) * np.random.randn(*x.shape)

# # 模擬參數
# n_samples = 10  # 軌跡數量
# n_steps = 100   # 時間步數
# T = 1.0         # 時間總長度
# g = 0.5         # 噪聲強度
# x_dim = 1       # 資料維度
# time = np.linspace(0, T, n_steps)
# dt = T / n_steps

# # 初始化資料分佈
# x0 = np.random.normal(size=(n_samples, x_dim))
# z0 = np.random.normal(size=(n_samples, x_dim))  # 加入 Z 軸初始值
# xt, zt = x0.copy(), z0.copy()

# # 儲存軌跡
# trajectories_forward_xz = [(xt, zt)]
# trajectories_reverse_xz = []

# # 正向SDE模擬 (同時計算 Z 軸的隨機擴散)
# for t in range(n_steps):
#     xt = forward_sde(xt, time[t], dt, g)
#     zt = forward_sde(zt, time[t], dt, g)  # Z 軸
#     trajectories_forward_xz.append((xt, zt))

# # 反向SDE模擬
# xt_reverse, zt_reverse = trajectories_forward_xz[-1]
# trajectories_reverse_xz.append((xt_reverse, zt_reverse))
# for t in reversed(range(n_steps)):
#     xt_reverse = reverse_sde(xt_reverse, time[t], dt, g)
#     zt_reverse = reverse_sde(zt_reverse, time[t], dt, g)  # Z 軸
#     trajectories_reverse_xz.append((xt_reverse, zt_reverse))

# # 轉換為陣列
# trajectories_forward_xz = np.array(trajectories_forward_xz)
# trajectories_reverse_xz = np.array(trajectories_reverse_xz)

# # 定義2D高斯分佈
# def gaussian_density(x, mean, cov):
#     """計算多變量高斯分佈密度"""
#     return multivariate_normal.pdf(x, mean=mean, cov=cov)

# # 建立2D背景 (XZ 平面)
# x_range = np.linspace(-3, 3, 100)
# z_range = np.linspace(-3, 3, 100)
# X, Z = np.meshgrid(x_range, z_range)
# pos = np.dstack((X, Z))
# mean = [0, 0]  # 高斯分布的平均值
# cov = [[1, 0], [0, 1]]  # 共變異數矩陣
# density = gaussian_density(pos, mean, cov)

# # 畫圖
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # 正向SDE圖 (XZ 平面)
# ax = axs[0]
# ax.contourf(X, Z, density, levels=50, cmap="viridis", alpha=0.7)  # 背景熱圖
# for i in range(n_samples):
#     ax.plot(trajectories_forward_xz[:, 0, i, 0], trajectories_forward_xz[:, 1, i, 0], color="red", alpha=0.6, label="Forward SDE" if i == 0 else None)
# ax.set_title("Forward SDE (XZ Plane)")
# ax.set_xlabel("X")
# ax.set_ylabel("Z")
# ax.grid()

# # 反向SDE圖 (XZ 平面)
# ax = axs[1]
# ax.contourf(X, Z, density, levels=50, cmap="viridis", alpha=0.7)  # 背景熱圖
# for i in range(n_samples):
#     ax.plot(trajectories_reverse_xz[:, 0, i, 0], trajectories_reverse_xz[:, 1, i, 0], color="blue", alpha=0.6, label="Reverse SDE" if i == 0 else None)
# ax.set_title("Reverse SDE (XZ Plane)")
# ax.set_xlabel("X")
# ax.set_ylabel("Z")
# ax.grid()

# # 儲存圖片
# plt.tight_layout()
# plt.savefig("sde_with_gaussian_background_xz.png", dpi=300)  # 儲存圖片
# plt.show()



