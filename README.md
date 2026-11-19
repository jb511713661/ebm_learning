# 冻结动作生成器 + 条件 EBM：CPU 教学实验

这是合成数据上的真实小模型训练，不是 π0.5/SmolVLA 或 Kaiwu 实测，不用于宣称机器人成功率提升。
先打开 `实验报告.html` 看图和结果，再看 `复现与学习.md`。源代码只有两个脚本。

## 运行

需要 Python 3.11 或更新版本。在本项目文件夹运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py
```

全部三组种子、100 轮在当前机器上只需数秒。实际速度因设备而异。
输出：`results/` 中的曲线、逐轮 CSV、汇总、参数检查点、推理样例；`data/` 中的可再生合成数据。
重新训练会覆盖同名输出。`--smoke` 仅供开发调试，会改为 5 轮和一个种子，也会覆盖输出；提交前重新执行完整命令。
HTML/Markdown 报告是此次实测快照；训练脚本更新 CSV 和图表但不自动重写报告中的数字。改变配置后，应同步更新报告。

## SwanLab（可选，联网上传）

```sh
python -m pip install -r requirements-swanlab.txt
swanlab login
python src/upload_swanlab.py --workspace YOUR_WORKSPACE --project ebm-learning-lab
```

只在官方登录提示中输入你自己的 Key，不粘贴到代码、聊天或报告。上传脚本读取已生成的真实 CSV，不生成虚构曲线；创建三个明确标注 synthetic 的实验。workspace 用你账户的实际空间名。项目是否公开需自行检查平台设置。

## 与原始材料的关系

借鉴会议“具身case 9.7”中的条件能量、BM/随机候选、冻结动作生成器和 list-wise 排序思想，以及用户提供的 `smolvla_ebm` 实现中的候选比较、softNCE 监督路径。
本项目从零写成，不要求安装原项目。Transformer 替换为固定二维条件特征，980 bit 缩成 6 bit，Kaiwu 模拟退火替换为枚举 64 个状态后的精确离散采样，动作模型替换为固定解析函数。不复刻论文的全部方法、复杂度和硬件路径。

## 5000 条扩展实验

```sh
python src/train.py --train-size 5000 --test-size 1000 --epochs 100 --output-dir runs/n5000
python src/upload_swanlab.py --run-dir runs/n5000 --project ebm-learning-lab
```

每个种子 5000 条训练、1000 条测试，共三组；新输出位于 runs/n5000，保留原报告基于 256 条数据的结果。原 HTML 报告不会自动更新。上传命令未指定 workspace 时使用 SDK 默认个人空间。若需团队空间，显式添加 --workspace 空间名。

## EBM 监控指标更新

每轮保存 CSV；SwanLab 上传以 epoch 为横轴，`train/optimizer_step` 记录真实累计参数更新次数。`train/grad_norm` 为本轮各批次更新前原始梯度 L2 范数的平均值（没有梯度裁剪）；`train/update_norm` 是 Adam 参数更新量 L2 范数的平均值；`train/parameter_norm` 为轮末参数范数；`train/lr` 是固定学习率。`train/target_entropy`、`train/kl_target_pred`、`train/regularization` 为各批次均值，满足 loss ≈ entropy + KL + regularization。KL 衡量同批候选中目标概率与预测概率的差异，候选随训练变化，不是固定测试集指标。`time/elapsed_seconds` 为本种子从开始训练计时的累计墙钟时间，不包含数据准备和最终输出写盘。轮 0 无更新，梯度和损失等为空，不伪造零值。

上传时：loss → train/loss；heldout_expected_mse → eval/bm_expected_mse；heldout_selected_mse → eval/selected_20_mse；random_baseline_mse → eval/random_one_mse。新增 eval/uniform_expected_mse 为精确均匀分布期望，可与 BM 单候选期望公平对比。旧日志仍可上传，但缺失的新监控指标无法追溯补出。上传发生在运行上传脚本时，不是实时云端记录。

```sh
python src/train.py --train-size 5000 --test-size 1000 --epochs 10000 --seed 7 --log-every 1 --checkpoint-every 1000 --output-dir runs/n5000_e10000_monitor_v2
python src/upload_swanlab.py --run-dir runs/n5000_e10000_monitor_v2 --project ebm-learning-lab
```

## Warm-up、能量与批次日志（v3）

新增可选 `--lr-schedule cosine --warmup-epochs 100 --min-lr-ratio 0.1`：按真实优化器更新步线性升至峰值，再余弦降至峰值的 10%。train/lr 为每轮最后一次更新实际使用值。默认 constant 保留原实验行为；旧实验无 warm-up 并非日志错误。

energy/all_states_mean：固定测试观察下全部 64 状态能量均值；within_context_std：先按观察计算状态间能量标准差再取均值；bm_expected：当前分布的期望能量；selected_20_mean：固定 20 候选中最低能量的均值；selected_gap：同组候选均值减最低能量。绝对能量有平移自由度，降低不等于动作质量改善；结合动作 MSE、KL 和能量差判断。

results/batches_SEED.csv 为更新前的真实批次 Loss、能量、梯度，lr 为该次更新实际使用值。默认每 50 次更新采点，可用 --batch-log-every 1 全部记录（10000 epoch 会产生约156万行，上传更慢）。每轮平均 CSV 保留。上传时生成两个独立实验：epoch 实验和 batches 实验，避免两种横轴混淆。批次波动来自实际抽样，未向指标添加随机抖动。合成规律仍然简单，不能声称达到真实 VLA 任务复杂度。

## 本次提交的实验版本

最新实验记录位于 `runs/n5000_e10000_v3/`：5000 条训练、1000 条测试、seed 7、10000 epoch，包含 warm-up、余弦衰减、能量监控、批次日志和每 1000 轮检查点。准确配置与数值以该目录的 config.json、summary.json 和 CSV 为准。根目录的 HTML/Markdown 报告及 results/ 是最初 256 条样本实验的历史快照，不能当作 v3 实验结果。

此仓库为独立教学实现，未 fork 上游 VLA 仓库；视频与原始 smolvla_ebm 代码仅为方法参考。未包含 π0.5、SmolVLA、LIBERO 权重或登录凭据。
