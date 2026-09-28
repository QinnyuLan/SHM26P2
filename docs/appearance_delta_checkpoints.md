# 严格的外观增量检查点

`src/bridge_rgs/checkpoints.py` 提供一种节省存储的显式格式
`bridge_rgs_appearance_delta_v1`。它只替换 `splats.sh0`、`splats.sh_rest` 和
`background_logits`，不是任意张量patch、增密结果或可直接strict-resume的模型。
基模型必须是保留在磁盘上的完整检查点，不能是另一个delta或已物化的delta。

## API与来源合同

```python
from bridge_rgs.checkpoints import APPEARANCE_KEYS, load_checkpoint, save_appearance_delta

receipt = save_appearance_delta(
    "runs/new_appearance/last.pt",
    base_checkpoint="runs/corner_v2_semantic_coupled/last.pt",
    base_sha256="a77d304f32de4356c1a4608ed5192258b6a9ced4c67a4297809f439481408d89",
    model={name: scene.state_dict()[name] for name in APPEARANCE_KEYS},
    manifest_path="artifacts/prepared_corner_v2/manifest.json",
    experiment="registered_appearance_arm",
    config=appearance_config,
    step=2000,
    trainer_state=None,
)
state = load_checkpoint("runs/new_appearance/last.pt", map_location="cpu")
```

`appearance_config`必须显式使用`parameter_scope: appearance_only`和固定pixel profile，
其manifest配置字符串与base一致。base必须有manifest SHA，新格式不靠今天的manifest
猜测旧模型profile。base缺profile仍按legacy处理，不能与v2 manifest组合。
保存时绑定manifest绝对路径/SHA、base绝对路径/调用者指定SHA；加载前后均核验依赖字节。

model恰好三个键；shape与base逐一一致，dtype一致且为dense floating tensor，全部有限。
另外核验SH张量形状与base高斯数量、SH阶数一致，background固定三个通道。
保存时先验证、再detach/CPU clone；张量view不会把不相关的巨大底层storage序列化。
文件通过同目录临时文件+fsync+原子hard-link发布；已有路径或并发出现同名结果均拒绝，
不覆盖旧结果。失败清理本次未发布临时文件，不触碰输入模型。

`load_checkpoint`对旧full checkpoint返回原有内容；delta会返回常规scene state字典：

- model除三个外观键外完全继承base；训练相机、架构、scene scale、base config及base step保留。
- base optimizers、torch/cuda/numpy RNG、density state、stats和未声明的训练状态不带入。
- `checkpoint_kind = appearance_delta_inference_or_warmstart`明确禁止普通strict resume。
- **新阶段**experiment/config/step存于`state["appearance_delta"]`，不能读顶层step当作新阶段步数。
- 可选独立trainer状态只存于`appearance_delta.trainer_state`，不提升为普通resume字段。
  允许的字段为optimizers、torch_rng、cuda_rng、numpy_rng、sampler_state、stats；它不自动
  支持续训。专用续训协议如未来需要，应另行实现与验证。
- `dependency_provenance`记录delta、base及manifest的绝对路径/SHA、固定profile及三个替换键。

模块本身不修改train/evaluate/official接口；调用者负责接入`load_checkpoint`并明确拒绝
普通resume。任何派生模型都不能被误记为不依赖base的完整产物，清理磁盘时必须保留base。
若原base config的manifest是相对路径，保存应从原工作区执行；载入依赖使用已记录绝对路径。

当前主源码已经由 `train.load_scene` 和 `official_evaluate` 接入此加载器，普通 train resume
会给出明确拒绝信息；render、native evaluate 和 PLY 沿用同一个 scene loader。
official评价与相机render回执记录delta的依赖来源，official评价末尾还会核验base/manifest未变。
新训练阶段步数仍只能读取命名空间中的字段；顶层step保持base阶段语义。

## CPU验证与预算

34项CPU回归通过，覆盖旧full逐字段兼容、两种profile、exact三键/shape/dtype/finite、
manifest/base变更、加载中base变更、禁止链式delta、旧训练状态剥离、独立trainer命名空间、
原子不覆盖/并发发布/磁盘写失败清理，以及实际GaussianScene的CPU strict state装载。
Ruff通过。没有GPU训练、没有改写现有checkpoint。

真实v2 base为240.295MiB；498,136个高斯的三外观FP32 tensor共91.211MiB，
其中sh0约5.701MiB、sh_rest约85.511MiB、背景12字节。
不带trainer state的delta约91.22MiB/臂；若同时保存两个同形状FP32 Adam moments，
主体约273.64MiB/臂，另有少量配置、step与RNG开销。
原子发布的hard-link不复制数据，但保存下一份文件时要同时容纳现存产物与临时文件；
当前接口不会覆盖同名last，正式runner应保存唯一固定终点或显式的新文件名。
