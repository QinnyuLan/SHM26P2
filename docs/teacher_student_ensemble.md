# 固定 50/50 的 DINOv3 与学生概率组合

这是较大模型、较高成本的工程对照。只执行一次预先固定的 0.5 概率平均，没有搜索权重；它不构成新机制，也不代表理论或性能上界。

student 使用 `runs/strong_semantic_averaged/last.pt`，teacher 使用 `runs/teacher_render_adapt_v1/best.pt`。两者共享 student 的单一 Gaussian geometry：只由相机参数渲染 RGB 与 student refined 概率，量化后的同一 RGB 进入 DINOv3，固定 tile768/stride512、水平翻转和 0.25 全景概率。所有预测完成后才读取 GT/valid 评分；未打开真实验证照片。全部 50 个相机均预测，其中 41 张标签计分。

执行时逐项确认了 teacher 与 student 的原有混淆矩阵完全复现、所有 RGB 像素与域适配基准相同、geometry/RGB 参数与 strongRGB 相同。运行源代码快照、输入 SHA、三个输出分支和结果位于 `runs/teacher_student_ensemble/`，主结果为 `metrics.json`。

| 分支 | all-5 mIoU | 前景 mIoU | 缆索 IoU | 缆索 2px F1 |
|---|---:|---:|---:|---:|
| Student | 93.9892% | 92.6805% | 93.7756% | 0.889949 |
| DINOv3 teacher | 94.3866% | 93.1552% | 93.9099% | 0.884676 |
| 固定 50/50 平均 | **94.7601%** | **93.6158%** | **94.4477%** | **0.909813** |

相对较高的单模型 mIoU，组合增加 **0.37349 个百分点**；相对 student 增加 **0.77091 个百分点**，但仍未达到 95%。RGB 沿用同一 strongRGB geometry：PSNR 30.2735 / SSIM 0.902818 / LPIPS 0.224494，组合不改变 RGB。这是当前插值开发集上的一次工程比较，没有新的独立测试集或多种子证据。

Student scene 共 **37,917,701** 个参数。DINOv3 teacher 共 **846,968,966** 个参数，其中冻结 backbone **840,592,640**，decoder **6,376,326**；共加载 **884,886,667** 个参数，Gaussian geometry 只保留一份。DINOv3 主干的高精度版本来自已固定 ModelScope H+/16，未用适配阶段 LoRA。

在与其他训练/评测并发的本次运行中，student 完整渲染平均 **0.0534 秒/视图**，DINOv3 后处理平均 **0.9490 秒/视图**，组合连同 RGB 来源审计约 **1.0547 秒/视图**；后者不包含模型加载和 GT 评分。当前进程 PyTorch 峰值已分配显存约 **3.374 GiB**。这些是并发条件下的测量，不是独占 GPU FPS，也不是统一部署接口的性能承诺。组合结果保留为独立对照，未接入默认轻量渲染流程。
