# NYU 随机权重推理测速

此入口只需推理依赖，不运行 `src/main.py` 的 Apex 训练链路；使用 `from_scratch`，不读取 ResNet 预训练文件。

## 环境与数据

从本仓库根目录执行：

```bash
conda env create -f environment.yaml
conda activate NLSPN
mkdir -p data
ln -s ../../data/nyudepthv2_h5 data/nyudepthv2_h5
```

已有环境或链接时不重复创建。`data/nyudepthv2_h5` 指向共享 HDF5 数据，需有 `nyu_h5_pairs.json`；搬到其他目录结构时仅调整软链接。`images/` 是插图目录，不是数据集目录。

## 测速与验证

```bash
python -m pytest tests/test_inference.py -q
python scripts/benchmark_nyu.py --output outputs/benchmark_nyu_new_run
```

输出目录必须尚不存在。默认使用真实 NYU 228×304、500 点、seed=2023、FP32、batch=1、关闭 TF32，三个样本各预热 100 次、同步计时 1000 次。只计预测前向，不计算 RMSE。

模型保持仓库默认 ResNet34、18 次传播、confidence propagation、TGASS，`legacy=False`、`preserve_input=False`。默认测速使用 torchvision DCN，保留 `legacy_dcn` 选项，但旧后端需另外编译扩展；不改结构或固定算子的初始化。

每次运行保存 `results.json`、`latency.csv`、GPU 状态、环境/源码快照和随机 FP32 state_dict。MACs 包含卷积（含转置卷积）及 DCN 名义累加，不含插值、调制和逐元素运算。随机权重结果不代表训练后精度或 checkpoint 对应的部署延迟。
