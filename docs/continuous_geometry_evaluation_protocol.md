# 连续 means 固定末态官方评价

本轮只评价已固定的 baseline、joint、pcgrad、trust 四个末态，不选择最好组、不适配 head、不增加采用门。baseline 是原 H3 `22bc8a…` 加完整 EM/FW 的固定实际 FP32 q，head 为原 H3。三候选仅替换各自 `means_final`；原 q 行序、颜色、尺度、不透明度、特征、head、相机均不变。训练完成且独立 CPU 审计 passed 后才能 prepare；任何失败保留，不重试。

完整继承 `direct_q_head_evaluation_recovery_v1` 冻结 package 与 direct-q adapter；与本轮训练的 model/refinement/depth-moments/coordinates/shader 源逐 SHA 核对。每个新 means 场重新生成自身 RGB、features、depth、alpha、moments，不能把旧上下文或旧教师软概率当新场输出。教师为 `00f5b8…` 的旧 H+ EMA，每张自身 legacy pinhole canvas RGB 经 clamp、乘 255、round 后实时推理：tile 768、stride 512、flip、context .25。教师 AMP 保留旧 BF16；全局评价为 cuDNN TF32=True、matmul TF32=False、highest、benchmark=False，沿用已证实的历史 E 评价设置，不冒充训练的 TF32=False。

固定原图 50 相机，其中 41 annotation；legacy overscan/K 和五通道 soft warp 复用官方函数。RGB 在浮点 canvas 上先回原图再 round；语义在 legacy canvas `(scene+teacher)*.5` 后一次 warp/argmax。baseline 的 canvas/原图 RGB 和 raw/scene/joint 共 150 mask 必须逐字节复现已完成 recovery `optimized_q_zero`；新教师 soft 须与该基线的旧教师输入结果 exact。在首次 GT payload 读取前，以此确认旧联合 mIoU `95.09720470648008%`。匹配旧预测是身份校验，不用旧预测替代本轮推理。

总计 200 scene、200 direct-q shader、200 head、200 `predict_image`，当前固定 canvas 每次 14 个 backbone forward（总 2,800）；gsplat 高层调用预期 400，低层按通道分块的 raster 调用另计，不混称 scene 数。保存 200 自身 RGB、600 mask 和 raw/scene/teacher FP32 soft 数组后，才读取 50 RGB GT 与 41 annotation，各一次；prepare 只继承 GT 预期 SHA，不读取/hash GT payload。评分沿冻结 `score_official_arrays`/LPIPS，200 次 RGB 评分在同臂三个读出间复用，产生 492 个语义 CM、12 份完整指标。只使用已缓存 LPIPS 权重并绑定哈希。

固定三候选各减 baseline 的 raw/scene/joint，共九组配对；另三组 joint 减 E。全部 5,000 次 paired bootstrap、seed 20260926。每臂指标里的 RGB 都是本轮自身 H3 RGB；E 是独立 1M+MCMC RGB 和既有语义的高成本工程参考，H3 means 变化不能称为 E RGB 的新收益。若将来另行组合 E RGB 与新语义，必须单独声明多场来源，不能在此混拼成绩。本轮无 best、无新性能门、无自动采用或创新声明。

内限 570 秒、外限 600 秒；既有四组 H+ 推理及多场评价约 267 秒、teacher 缓存版 direct-q 恢复约 115 秒只作成本依据，不保证新运行成功。记录全部来源、运行模块、renderer binary、teacher/backbone/LPIPS 依赖，恢复 means、flags、模式、numerics；RustDesk 仅沿既有真实 executable 例外，不终止用户进程。计时包括真实 H3 渲染、教师、数组写盘与评分，不包含 E 的两 RGB 场渲染，不称完整 E 部署 FPS。
