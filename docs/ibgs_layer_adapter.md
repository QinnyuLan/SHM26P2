# 冻结场 live-buffer 接线

`ibgs_layer_adapter.py` 是一个薄前向入口，不修改原模型/backend。`load_fixed_full(path)` 只接受旧 full 6000 端点 `e977dc…`，经已有 `load_saved_field` 载入后冻结全部8组 field 参数及背景；不重新初始化 normal/offset，不反传场。`load_selector_extension(path, expected_sha256)` 只载入检查通过后已生成的明确 `.so` 路径/哈希，不触发 JIT。调用者仍须绑定原 snapshot、原后端 `436b2b…` 与 selector passed/natural-completion 回执；不是此小函数从哈希推断数值通过。

`render_layer_capture(camera, field, background, extension=...)` 接收单独创建的 corner-v2 `BridgeCamera`，nearest 两列表必须为空。原 `render(...render_geo=True, return_depth_normal=False)` 只执行一次；临时观察 `_C.rasterize_gaussians`，原29输入/13输出不改。保持原CUDA buffers存储引用，在 renderer 返回后按它们**当前真实 data_ptr**调用 `buffer_views`，再一次 selector 同时得到 median4/top4。输入置于 `torch.no_grad()`，而非 inference tensor 模式，使下游 trainable head 可安全保存常量输入；始终 finally 恢复原 backend 函数。无需第二次目标 render，无 RGB source bank/depth render，无 backward。

返回 `raw[3,H,W]`、`ray[3,H,W]`、`median4/top4` 各 `ids/depth/weights/ordinals[H,W,4]`、`status[H,W]` 与轻量 metadata。metadata含实际FP32 world_to_camera/camera_center、focal/principal、原buffer地址及单次计数。原scratch buffers在返回后可释放。回调依赖注入仅用于CPU合成接线测试；真实调用者不得因此绕过来源绑定。

缓存 worker 可以对350TRAIN各调用一次，仅保留 source top4 ID/w/z 和同一 eroded-valid；新连续support不使用单median门，故无需额外350次scalar depth render。CPU存350份ID/w/z约20.426 GiB，加uint8 RGB约1.277 GiB；每步四源转GPU约0.292 GiB。目标16×7特征约0.545 GiB，MLP每个完整32通道隐藏激活约2.49 GiB，仍须真实单步预检峰值，不能按参数量宣称原方法同算力。source RGB依旧来自TRAIN，目标相机不需要目标照片进行推理。

现三对照共享固定场、相同新head初始化/步数/顺序/q：median4-mass、top4-mass、top4-normalized。归一化仅是 pooled evidence 的同分母控制，不能同时变源支持、图像采样或网络容量；原空支持像素均精确base。模块不选择训练预算或自动采用任何末态。
