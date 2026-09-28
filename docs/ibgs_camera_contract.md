# corner-v2 → 官方 IBGS 的相机与像素合同

2026-09-27。只读核官方仓库 `977e96c6574f20b456c760942e615031154b4cd5` 的相机、投影、前向／反向与采样；没有改第三方源、编译、GPU 或读取真实图像。独立模块 `src/bridge_rgs/ibgs_camera_contract.py` 仅提供CPU代数转换。官方原数值backend smoke与未来坐标修订必须分开记录，不能把修订后结果称原实现逐位复现。

## 当前源码存在半像素不一致

[auxiliary.h:45](/mnt/data/SHM2026/third_party/ibgs/submodules/diff-plane-rasterization/cuda_rasterizer/auxiliary.h:45) 的 `ndc2Pix(v,S)=((v+1)*S−1)/2` 将NDC零映到 `(S−1)/2`。默认居中投影使高斯中心为 `fx*X/Z + W/2−.5`，正好等于 corner-v2 投影减 `.5` 后的整数数组坐标。CUDA的 `pixf=(j,i)` 与该中心一致，**不要将 pixf 改成 j+.5，也不要改 ndc2Pix**。

但 [rasterizer_impl.cu:477](/mnt/data/SHM2026/third_party/ibgs/submodules/diff-plane-rasterization/cuda_rasterizer/rasterizer_impl.cu:477) 将 `W/2,H/2` 传给平面交点射线，[forward.cu:352](/mnt/data/SHM2026/third_party/ibgs/submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu:352) 使用 `(j−cx)/fx`；[backward.cu:545](/mnt/data/SHM2026/third_party/ibgs/submodules/diff-plane-rasterization/cuda_rasterizer/backward.cu:545) 独立重复了同一主点。[Camera](/mnt/data/SHM2026/third_party/ibgs/scene/cameras.py:71) 的 `Cx,Cy/get_rays/get_k` 也用 `W/2,H/2`。因此交点射线比同一光栅像素偏移半像素。同相机错误逆投影＋错误投影恰可抵消，不能仅以 identity warp 证明整链正确；测试增加源相机z平移，明确检出该差异。

## 完整转换

本项目实际送入渲染的无畸变 corner K、w2c 数值提升FP64作合同检查。设 `Kc=(fx,fy,cx,cy)`，像素为角点坐标 `(j+.5,i+.5)`。IBGS内部数组K必须是 `Ka`，只将主点改成 `cx−.5,cy−.5`：

`r_cam=((j−Ka.cx)/fx,(i−Ka.cy)/fy,1)`。

相机z深度乘此未归一化射线；不能乘单位长度射线。源相机点由 `Ts @ inv(Tref)` 转换（列向量）。源数组投影为 `(fx*Xs/Zs+Ka.cx,fy*Ys/Zs+Ka.cy)`。

高斯NDC投影则仍用 **corner K**：列向量投影矩阵的 `P00=2fx/W, P11=2fy/H, P02=2cx/W−1, P12=2cy/H−1, P32=1`；near/far项为官方正z式。故 `ndc2Pix(Px/Pw)=fxX/Z+cx−.5`。给 `getProjectionMatrixCenterShift` 的必须是corner主点，不能传Ka再减第二次 `.5`。当前1M场是centered corner K，默认对称FoV投影已经正确；无需改变它。

官方 Camera 的 `R` 参数是 **w2c旋转的转置**，`T` 参数是w2c平移。`getWorld2View2` 再转置R生成w2c；Camera保存 `world_view_transform=w2c.T` 供GLM列主序解释，`projection_matrix=P.T`，`full_proj_transform=w2c.T@P.T`。camera center为 `inv(w2c)[:3,3]`。不得额外左右手翻转、交换相机到世界／世界到相机，且本项目保持 `trans=0,scale=1`。`scene.world_view_transforms` 再转置回真正w2c，renderer的 `world_to_source @ inverse(world_to_reference)` 正确。

## 源图像采样和Python几何辅助

主IBGS源颜色／深度采样使用CUDA texture，**不是**PyTorch grid_sample。[forward.cu:550](/mnt/data/SHM2026/third_party/ibgs/submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu:550) 在数组投影上加 `.5` 才传 `tex2DLayered`；texture设为非归一化坐标、linear filter。这是数组中心→texture中心的必需转换，修主点后必须保留。backward的texture采样同样保留加 `.5`。

Python [get_points_depth_in_depth_map](/mnt/data/SHM2026/third_party/ibgs/scene/gaussian_model.py:605) 用Ka式投影、`2*x/(W−1)−1` 和 `align_corners=True`，在scale=1时一致。若改用 `align_corners=False`，必须配 `2*(x+.5)/W−1`，不能仅切flag。`graphics_utils.depth_pcd2normal(offset!=None)` 当前用前一种grid却省略align_corners，属于可选路径的独立风险；最小端口应不启offset或明确改成True。网络图像resize的align_corners不等同相机投影，不应全仓替换。

缩放分两种：图像resize先按实际 `Wnew/Wold,Hnew/Hold` 缩corner K，再转Ka；例如989→494不能把高度比例写成精确1/2。整数stride抽样 `start+step*j` 的Ka主点则是 `(Ka.cx−start)/step`，不是resize公式。官方若调用scale>1的depth stride路径，要按实际start同步get_rays/get_k/grid；首轮最小端口固定native、scale=1可避开这项额外修改，不能暗称其它比例已经验证。

## 最小官方补丁建议（本次未应用）

1. `scene/cameras.py`：在目标／源共同native尺寸，令Python数组主点 `Cx=(W−1)/2, Cy=(H−1)/2`；FoV和对称projection保持原值。加载caller从项目w2c传 `R=w2c[:3,:3].T,T=w2c[:3,3]`，原prepared_corner_v2图像不再畸变或shift。
2. `cuda_rasterizer/rasterizer_impl.cu` 的 `FORWARD::render` 调用：传 `(width−1)*.5f,(height−1)*.5f`。
3. `cuda_rasterizer/backward.cu`：先定义 `cx=(W−1)*.5f,cy=(H−1)*.5f`，**单独那行 ray 也必须改为使用cx/cy**。其后的交点、源投影和 `dp/dd` 都沿这两个变量。只改cx/cy、漏改前一行ray是不完整修复。
4. 保留整数pixf、ndc2Pix、投影FoV、texture坐标+.5、纹理过滤方式及既有backward算法；不得用相机平移补偿像素偏移。

官方CUDA源投影复用目标的fx/fy/cx/cy、W/H，没有每源intrinsics参数。因此最小端口强制所有source/target相同K和尺寸，且corner主点正好W/2,H/2。CPU模块虽能代数表示非中心K，但 `require_official_shared_centered` 会拒绝它及异源K；不能仅改Python投影后宣称CUDA支持任意内参。若以后需要这些输入，必须扩展完整CUDA接口与反向，而非本次半像素补丁。

CPU测试覆盖整链射线↔NDC↔数组、非identity跨相机反例、Camera R转置／相对变换、纹理坐标、两种grid约定的真实CPU grid_sample、奇数高度resize/stride区别、非中心／异源K拒绝，以及直接调用官方Python投影函数核一致。官方preprocess另有 `1/(clip_w+1e−7)`，这是独立微小深度相关偏差，测试显式区分，不为代数恒等式悄悄修改CUDA。8项通过只证明这些CPU公式／域合同，不证明两个不同光栅器RGB、抗锯齿、排序、梯度或性能逐位相同。
