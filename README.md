# Grammar Variational Autoencoder

**grammarVAE** 是一个用于将 SMILES 描述符（用于表示分子的线性文本格式）进行变分自编码器（VAE）编码的项目。该项目还扩展到处理数学方程，提供了对分子和方程的生成、编码/解码以及优化功能。项目包含数据处理、模型训练、采样和贝叶斯优化等多个模块。本项目在原作者项目基础上进行算法优化

# 一.项目说明

## 目录结构及说明

```
grammarVAE_train
├─ README.md
├─ make_zinc_dataset_grammar.py
├─ models
│  ├─ __init__.py
│  ├─ model_zinc.py
│  └─ utils.py
├─ molecule_vae.py
├─ requirements.txt
├─ train_zinc.py
├─ zinc_grammar.py
└─ zinc_grammar_dataset.h5
```

# 二.项目复现指南

为了有效地使用**grammarVAE**项目进行分子训练和优化，需按照以下步骤依次运行相关文件。每一步的文件及其使用场景如下：

### 步骤1：环境配置

1. **安装依赖**

- **文件**: requirements.txt
- **使用场景**: 配置项目所需的Python环境。
- **操作**:

```bash
pip install -r requirements.txt
```

**注意**: 如果需要GPU加速，按照README.md中的说明替换TensorFlow GPU版本的链接。

### 步骤2：数据准备

1. **生成基于语法的数据集**

- **文件**: make_zinc_dataset_grammar.py
- **使用场景**: 预处理原始SMILES数据，生成适用于基于语法的VAE模型的数据集。
- **操作**:

```bash
python make_zinc_dataset_grammar.py
```

**输出**: 生成data/eq2_grammar_dataset.h5等处理后的数据文件。

2. **生成基于字符串的数据集**

- **文件**: make_zinc_dataset_str.py
- **使用场景**: 预处理原始SMILES数据，生成适用于基于字符串的VAE模型的数据集。
- **操作**:

```bash
python make_zinc_dataset_str.py
```

**输出**: 生成data/eq2_str_dataset.h5等处理后的数据文件。

### 步骤3：模型训练

1. **训练基于语法的VAE模型**

- **文件**: train_zinc.py
- **使用场景**: 使用基于语法的数据集训练VAE模型。
- **操作**:

```bash
python train_zinc.py
```

**可选**: 使用自定义参数，如更改潜在空间维度或训练轮数。

```bash
python train_zinc.py --latent_dim=2 --epochs=50
```

**输出**: 训练好的模型权重保存在models/model_zinc.hdf5（具体文件名视脚本实现而定）。

2. **训练基于字符串的VAE模型**

- **文件**: train_zinc_str.py
- **使用场景**: 使用基于字符串的数据集训练VAE模型。
- **操作**:

```bash
python train_zinc_str.py
```

**可选**: 使用自定义参数，如更改潜在空间维度或训练轮数。

```bash
python train_zinc_str.py --latent_dim=2 --epochs=50
```

**输出**: 训练好的模型权重保存在models/model_zinc_str.hdf5（具体文件名视脚本实现而定）。

### 步骤4：编码与解码

1. **编码和解码SMILES字符串（演示）**

- **文件**: encode_decode_zinc.py
- **使用场景**: 演示如何使用训练好的VAE模型对SMILES字符串进行编码和解码。
- **操作**:

```bash
python encode_decode_zinc.py
```

**流程**:加载预训练或训练好的VAE模型。读取示例SMILES字符串。编码为潜在向量。解码回SMILES字符串。显示结果。

### 步骤5：分子优化

1. **生成潜在特征和目标属性**

- **文件**: `molecule_optimization/latent_features_and_targets_character/generate_latent_features_and_targets.py`
- **使用场景**: 使用基于字符的VAE模型生成分子的潜在向量和目标属性（如SA score）。
- **操作**:

```bash
cd molecule_optimization/latent_features_and_targets_character/
python generate_latent_features_and_targets.py
```

**输出**: 生成`latent_features.pkl`（示例名称），包含潜在向量和SA score。

- **文件**: `molecule_optimization/latent_features_and_targets_grammar/generate_latent_features_and_targets.py`
- **使用场景**: 使用基于语法的VAE模型生成分子的潜在向量和目标属性。
- **操作**:

```bash
cd molecule_optimization/latent_features_and_targets_grammar/
python generate_latent_features_and_targets.py
```

**输出**: 生成`latent_features.pkl`（示例名称），包含潜在向量和SA score。

### 步骤6：运行贝叶斯优化模拟

1. **运行贝叶斯优化**

- **文件**: `molecule_optimization/simulationX/character/run_bo.py` 和 `molecule_optimization/simulationX/grammar/run_bo.py`（X=1至10）
- **使用场景**: 在潜在空间中执行贝叶斯优化，搜索具有最佳目标属性的分子。
- **操作**:进入每个模拟目录的character和grammar子目录，分别运行优化脚本。**示例**:

```bash
cd molecule_optimization/simulation1/character/
python run_bo.py

cd ../grammar/
python run_bo.py
```

**并行运行**: 为了节省时间，建议在不同的终端或使用作业调度系统（如nohup）并行运行多个优化任务。

```bash
nohup python run_bo.py &
```

**优化流程**:加载潜在向量和目标属性。初始化高斯过程模型（通过gauss.py或sparse_gp.py）。迭代执行：选择下一个采样点（潜在向量）。解码生成SMILES字符串。计算目标属性（如SA score）。更新高斯过程模型。保存结果至results/dummy_file.txt。

### 步骤7：汇总和分析结果

1. **提取和汇总最终结果**

- **文件**: molecule_optimization/get_final_results.py
- **使用场景**: 从所有模拟实验中提取和汇总最佳分子及其属性，生成综合报告。
- **操作**:

```bash
cd molecule_optimization/
python get_final_results.py
```

**输出**: 生成final_results.pkl（示例名称），包含所有模拟的最佳分子信息。

2. **计算和汇总平均性能指标**

- **文件**: `molecule_optimization/get_average_test_RMSE_LL.sh`
- **使用场景**: 计算所有模拟实验的平均测试均方根误差（RMSE）和对数似然（LL）。
- **操作**:

```bash
./get_average_test_RMSE_LL.sh
```

**注意**: 确保脚本具有执行权限。

```bash
chmod +x get_average_test_RMSE_LL.sh
```

**输出**:

```php
Average Test RMSE: 
Average Test LL: 
```

### 步骤8：查看优化结果

1. **查看最佳分子图像**

- **目录**: molecule_optimization/molecule_images/
- **文件**: best_character_molecule.svg 和 best_grammar_molecule.svg
- **使用场景**: 直观查看优化过程中找到的最佳分子结构。
- **操作**:

```bash
cd molecule_optimization/molecule_images/
ls
```

使用SVG查看器（如浏览器）打开这些文件，查看最佳分子的结构。

### 步骤9：使用预训练模型（可选）

1. **使用预训练模型进行编码和解码**

- **文件**: encode_decode_zinc.py
- **使用场景**: 使用预训练的VAE模型快速进行编码和解码，而无需重新训练模型。
- **操作**:

```bash
python encode_decode_zinc.py
```

**流程**:加载预训练的VAE模型（如pretrained/zinc_vae_grammar_L56_E100_val.hdf5）。读取示例SMILES字符串。编码为潜在向量。解码回SMILES字符串。显示结果。

### 步骤10：总结与验证

1. **验证项目运行**

- **操作**:确保所有步骤按顺序无误地完成。检查生成的模型权重和结果文件是否存在且正确。通过encode_decode_zinc.py确认模型的编码和解码功能正常。
- **验证示例**:

```bash
python encode_decode_zinc.py
```

**预期结果**:

```yaml
Original SMILES: CCO
Encoded Latent Vector: [ ... ]
Decoded SMILES: CCO
```

## 三、总结

通过上述步骤和详细说明，你可以系统地使用**grammarVAE**项目进行分子VAE模型的训练和优化。以下是关键点的总结：

1. **环境配置**: 安装所需的Python库和依赖，确保项目正常运行。
2. **数据预处理**: 生成适用于不同VAE模型的数据集，确保数据质量和格式符合要求。
3. **模型训练**: 训练基于语法和基于字符串的VAE模型，生成高质量的潜在空间表示。
4. **编码与解码**: 验证模型的编码和解码功能，确保模型能够准确地转换SMILES字符串和潜在向量。
5. **分子优化**: 使用贝叶斯优化在潜在空间中搜索最佳分子，提升分子属性（如SA score）。
6. **结果汇总**: 汇总和分析优化结果，评估优化算法的有效性和稳定性。
7. **使用预训练模型**: 在需要时，使用预训练模型进行快速编码和解码操作，节省训练时间。
   通过系统地遵循上述步骤，你可以充分利用**grammarVAE**项目进行分子生成和优化，推动药物发现和材料科学等领域的研究进展。如果在使用过程中遇到任何问题，建议参考项目的源代码和文档，或在相关社区寻求帮助。

# 项目优化方案

- [x] 项目支持分子数据集和方程数据集,只保留分子训练的部分，删除公式训练部分文件
- [x] python2.7迁移至python3环境
- [ ] 优化项目依赖包版本,迁移原项目开发环境至Python 3.10 + TF2.18 + Keras3.7 + CUDA加速
- [ ] 整合该项目到DLEPS项目中
- [ ] 优化项目代码结构，将相关模块进行封装，提高代码可读性、可维护性
