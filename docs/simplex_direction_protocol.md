# 直接类别概率端点的一次方向核验

这是独立于已完成 `raw_simplex_fullbatch_v1` 的数值诊断，不修改其一次接受、八次回溯耗尽的结果，不恢复训练，不使用 VAL/teacher/head。标准 FW 顶点与一维凸线搜索不是学术创新。旧端点含许多精确零概率，可能造成噪声地板附近的大 CE 曲率，但既有记录尚不能排除 VJP 误差。

本次执行协议为 `simplex_direction_diagnostic_v2`。v1 的冻结 CPU 测试误启 pytest cacheprovider，在快照中新增4个 `.pytest_cache` 文件；绑定源无修改或缺失。worker 在任何 GPU/渲染/标签解码前拒绝 source inventory，全部计数为0，原失败目录完整保留。v2 绑定该失败 plan/执行回执/launch/旧入口和新增缓存 SHA；仍从原 `raw_simplex_fullbatch_v1` 干净快照继承源。唯一修正为协议/失败来源绑定与测试流程：冻结测试使用 `-p no:cacheprovider`，前后逐项核源 SHA。下文数学、阈值、64次二分及时间/次数预算均与 v1 相同，不是数值失败后的调参或重试。

固定旧 endpoint q、H3/legacy、259 TRAIN、同 class weights/valid/仿射噪声 δ=5e−7。继承旧冻结包和已通过的 direct-q 预检接口；全部模型/相机冻结。最多三次完整 pass：

1. 当前 q 的完整目标及梯度。逐 view FP32 VJP 转 FP64 累加再除259；保存各 view 的 g32、有效 GT 类别和原始 target raw `a`。不提前按259缩小 VJP。
2. 唯一 LMO 顶点 v（逐行 argmin、平票最低类别）的前向，保存 target raw `b_endpoint`。此后固定方向，不换为 PG 或挑方向。由独立端点先升 FP64 再相减建立 `p(t)=(1−δ)[a+t(b_endpoint−a)]+δ/5`；同一个残余背景在差值中抵消，但保留在分母和目标里。
3. 只有下述方向门通过，沿这一固定缓存凸线段做64次导数二分；导数在1仍非正则取1。master 候选 `(1−γ)q_master+γv` 只 cast FP32，不修补。仅一次真实全量前向确认，绝不按失败结果调整 γ 或换方向。若cast不变，报告浮点停滞，不做无意义第三pass。

核三量：`A=mean_view <g32_view, v32−q32>`；`B64=mean_view mean_valid[−w(1−δ)Δraw/p(0)]`；`B32upstream` 将每view的 `−w(1−δ)/(Nvalid*p(0))` 先castFP32再以FP64与Δraw点积。A–B32、B32–B64分别需满足 `abs误差 <= 2e−7 + 5e−4*max(abs两量)`；三个量均为负且绝对值均超过 `10*2e−7`。这是执行前固定、沿既有线性伴随校准尺度的工程数值门，不是严格 FP32 区间界。失败或信号不足即停止，不执行第三pass。

固定概率区间 `[0,1e−6,1e−5,1e−4,1e−3,1e−2,.1,∞)` 仅分解 B64 的正/负/绝对贡献与像素数，并报告取消比 `Σ|Bpixel|/|B|`。没有分箱 VJP，不声称已定位每箱梯度错误。保留逐view A/B 三量供独立核验。真实候选 F 必须严格下降，且缓存预测下降与实测下降之差使用同一 `2e−7+5e−4*max(abs下降)` 门；完成不等于采用/收敛。

全部缓存与产物在 `/mnt/data`。现有形状上界为338,119,320像素、498,136 Gaussian；两组target FP32为2.705GB，labels uint8 .338GB，259份g32为2.580GB，合计5.624GB。逐262144像素chunk做FP64运算，mmap逐view梯度，预计额外内存低于16GiB；prepare检查可用RAM≥16GiB及磁盘预算。2026-09-27准备前只读检查RAM available59.96GB、数据盘余935GB，非运行峰值证明。内600秒/外660秒覆盖加载、三pass、缓存、标量搜索、hash和恢复；时间到保留失败，不延长/重试。独占 started 标记在GPU工作前创建。

只有证据通过才能说“该固定端点、固定FW方向的当前数值链与一次下降一致”；不能证明所有方向VJP正确、原生属性充分优化、IoU最优或几何表示不足，更不能称改善新视图性能。
