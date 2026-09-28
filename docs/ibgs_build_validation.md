# IBGS 独立构建验证

2026-09-27：官方 IBGS 的两个 CUDA 扩展在独立 Python 3.11 / PyTorch 2.8.0+cu128 环境中构建、导入成功，二进制均包含 `sm_120`。本报告只证明构建和导入；没有执行 GPU 前向、训练或数据像素读取。后续 GPU 烟测由根代理单独记录，不能由本报告推定通过。

来源为 [作者仓库](https://github.com/HoangChuongNguyen/ibgs) 的固定 commit `977e96c6574f20b456c760942e615031154b4cd5`，本地 `/mnt/data/SHM2026/third_party/ibgs`。原许可证和版权头保留；本次本地验证不代表已取得发布或商业使用许可。集成边界见 [可行性核查](ibgs_integration_feasibility.md)。

独立解释器为 `/mnt/data/SHM2026/third_party/ibgs/.venv/bin/python`，`PYTHONPATH=/mnt/data/SHM2026/third_party/ibgs`。使用 torchvision `0.23.0+cu128`、CUDA toolkit `12.8.93`、`MAX_JOBS=4`、`TORCH_CUDA_ARCH_LIST=12.0`、`CUDA_HOME=/usr/local/cuda`。所有新环境、缓存、临时编译文件和日志均放数据盘；复用主环境已有 wheel 缓存的只读副本。主项目 `uv.lock` 与 `.venv/pyvenv.cfg` 前后 SHA 不变，没有对主环境执行安装。

每次构建独立计时并保留日志，子进程组 590 秒超时后先终止、至多再等 5 秒。首次失败没有覆盖或追认：

| 构建尝试 | 实际耗时 | 结果 |
|---|---:|---|
| 原 diff-plane-rasterization | 14.889 s | 编译失败：缺失 `<cstdint>`，`std::uintptr_t` 等类型未声明 |
| 补 `<cstdint>` 后 | 21.500 s | 自然 exit 0 |
| 原 simple-knn | 21.205 s | 编译失败：缺失 `<cfloat>`，`FLT_MAX` 未声明 |
| 补 `<cfloat>` 后 | 8.979 s | 自然 exit 0 |
| 两扩展纯导入 | 2.818 s | 自然 exit 0，CUDA 未初始化 |
| GaussianModel / gaussian_renderer 纯导入 | 1.466 s | 自然 exit 0，CUDA 未初始化 |

兼容修改只有上述两个标准头，以及 `scene/gaussian_model.py` 将 PyTorch3D 的 `quaternion_to_matrix` 别名替换为项目已有 `build_rotation`。两处实际调用均为 CUDA FP32、Nx4、scalar-first 四元数。CPU 合同提取原函数 AST，仅将其硬编码 CUDA 分配重定向到 CPU，独立 NumPy FP64 公式核验 2 个恒等、4096 个单位与 4096 个非单位四元数；最大绝对差 `4.5102e-7`，通过预定 `8e-7` 容差。这不是对任意 dtype/device/零四元数的 PyTorch3D 通用替代保证，也未证明梯度逐位相同。数值报告从保留的测试 stdout 提取；原同名 JSON 被外层运行回执占用，stdout 与回执均保留，没有重跑测试。

构建工具另自动改写两个已跟踪 `egg-info/PKG-INFO` 文件，KNN 源文件补末尾换行；这些也完整记录在 patch 中。**没有应用半像素、投影、着色器数值或训练配方修改。**

| 安装模块（均在新 venv 的 `lib/python3.11/site-packages`） | 二进制 SHA256 |
|---|---|
| `diff_plane_rasterization/_C.cpython-311-x86_64-linux-gnu.so` | `de486f08b6f6759e339f3bb833202e218fa1432960c54a303516787b26976d0d` |
| `simple_knn/_C.cpython-311-x86_64-linux-gnu.so` | `2bfc7f1b145fe8a91fe5f63327a49bb662687d0749014677f1d01142b48d8f2f` |

两个原数值二进制另保存到验证目录，避免后续坐标修订覆盖来源。除构建依赖外仅安装实际导入需要的 `plyfile`、`matplotlib`、`opencv-python`，没有安装 PyTorch3D/open3d。当前独立 OpenCV 为 `5.0.0.93`，不冒称与主项目版本相同；完整依赖已经冻结，`uv pip check` 通过。`render.py` 可选轨迹代码仍引用 PyTorch3D，不属于已验证入口。

全部证据在 [`ibgs_build_validation_v1`](/mnt/data/SHM2026/third_party/ibgs_build_validation_v1)：[总回执](/mnt/data/SHM2026/third_party/ibgs_build_validation_v1/build_validation_report.json) SHA `1c051fa26900da53f986d7640b57164a1713753ea19972a41e7595e686fb2f4e`；[依赖冻结](/mnt/data/SHM2026/third_party/ibgs_build_validation_v1/requirements_frozen.txt) SHA `3afd06d3ed2010fadb829d990c165df93670033a9dcd2c2d2ec09f7c0309b8b5`；[完整兼容差异](/mnt/data/SHM2026/third_party/ibgs_build_validation_v1/compatibility.patch)。总回执绑定 1654 个 tracked source 文件的 SHA、各失败/成功日志、独立环境及二进制。数据盘结束时剩余约 813 GiB；实际 GPU 显存与运行速度尚未在本任务测量。
