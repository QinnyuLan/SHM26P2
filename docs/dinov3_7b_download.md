# DINOv3 ViT-7B 的固定来源下载

独立入口 `scripts/download_dinov3_7b.py` 只下载 ModelScope 的
`facebook/dinov3-vit7b16-pretrain-lvd1689m`，revision 固定为
`5251e00b307184bb247d076713375235d554fefb`。脚本内锁定 12 个文件的大小与 SHA256：
config、ModelScope configuration、preprocessor、license、README、safetensors index，以及六个权重分片。
文件清单必须与固定 revision 的服务器声明相同，传输后必须再通过本地完整字节 SHA。

```bash
uv run --no-sync python -u scripts/download_dinov3_7b.py --workers 2 --attempts 3
```

默认目录为 `/mnt/data/SHM2026/models/dinov3-vit7b16`。CLI 拒绝将输出放到 `/mnt/data` 之外。
没有使用 ModelScope/HF 公共下载缓存，也没有第二份完整权重副本；`.part`、进度、锁文件和最终文件都在目标目录。
每个分片先写 `filename.part`，通过大小和 SHA 校验后在同目录原子 rename 为最终名称。

如果连接中断，脚本每文件最多尝试三次；失败保留已下载字节，退出失败状态。
之后执行同一命令会重新核验已经完成的文件，并用 `Range: bytes=<partial-size>-` 续传尚未完成的分片。
服务器必须返回完全匹配的 Content-Range；若忽略续传 Range，脚本直接拒绝，绝不把完整响应追加到部分文件后面。
完成但 SHA 错误的文件也保留并报错，不自动删除、覆盖或无限重新下载。
目录内排他文件锁拒绝两个下载器同时写同一份分片。

运行状态写在 `download_state.json`，固定声明写在 `download_source_manifest.json`。
只有全部文件通过校验且 index 恰好引用六个固定分片时，才产生 `download_provenance.json` 的
`status: completed`。该 receipt 同时记录 provider/model_id/revision/model_dir、下载入口源码 SHA、
源清单 SHA、每个文件的服务器声明和实际 SHA，以及 `files_sha256` 便于消费者校验。
消费者不能以 `.part` 的存在或某个分片完成作为整个模型就绪的依据。

可在下载完成、无下载器活动后再次执行：

```bash
uv run --no-sync python scripts/download_dinov3_7b.py --verify-only
```

这一模式重新核实固定 revision 的服务器声明和 12 个本地文件 SHA，不加载模型、不使用 GPU。
不需要重新安装依赖。原 `scripts/download_dinov3.py` 和 H+ 模型目录没有修改。

`tests/test_download_dinov3_7b.py` 包含 8 个 CPU/无网络检查：锁定清单、真实字节模拟的断点续传、
已有 partial 重启、服务器忽略 Range 拒绝、SHA 错误保留、有效文件跳过下载、三次重试上限、
12 文件声明与前期 ModelScope 只读审查一致。启动前全部通过，Ruff 通过。
