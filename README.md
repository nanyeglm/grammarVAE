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

## 二、核心文件及其作用

以下是项目中与分子训练相关的核心文件和目录，以及它们的具体作用：

### 1. **`data` 目录**

- `250k_rndm_zinc_drugs_clean.smi`:包含250,000个经过清洗的ZINC分子库中的SMILES字符串。

### 2. **`models` 目录**

 存放VAE模型的定义和相关工具。

- **`model_zinc.py`**: 定义基于语法（Grammar-based）的ZINC分子VAE模型。包含VAE模型的架构定义（编码器和解码器）、损失函数等。

- **`model_zinc_str.py`**: 定义基于字符串（String-based）的ZINC分子VAE模型。 类似于`model_zinc.py`，但不使用语法规则，而是直接基于字符进行编码和解码。

- **`utils.py`**:  包含辅助函数，如数据预处理、模型加载与保存、评估指标计算等。

### 3. **`molecule_optimization` 目录**

 负责分子的优化过程，尤其是通过贝叶斯优化在潜在空间中搜索最佳分子。

#### 子目录和文件详解

##### a. **`latent_features_and_targets_character`** 和 **`latent_features_and_targets_grammar`**

 这两个子目录分别对应于基于字符和基于语法的VAE模型生成的潜在特征和目标属性。

- **`fpscores.pkl.gz`**: 压缩的Pickle文件，包含分子的指纹评分（Fingerprint Scores）。在优化过程中，用于评估分子的属性，如合成可行性（SA score）。

- **`generate_latent_features_and_targets.py`**
       1. 加载分子数据集（SMILES字符串）。
       2. 使用训练好的VAE模型将分子编码为潜在向量。
       3. 计算分子的目标属性（如SA score）。
       4. 保存潜在向量和目标属性，供贝叶斯优化使用。

- **`sascorer.py`**:用于计算分子的合成可行性评分（SA score）。 提供计算分子SA score的函数，评分越高，合成难度越大。

##### b. **`molecule_images`**

 存储优化过程中发现的最佳分子的图像表示。

- **`best_character_molecule.svg`**: 使用基于字符的VAE模型在优化过程中找到的最佳分子的SVG图像。

- **`best_grammar_molecule.svg`**: 使用基于语法的VAE模型在优化过程中找到的最佳分子的SVG图像。

##### c. **`simulation1` 至 `simulation10`**

 这些子目录代表了10次独立的贝叶斯优化实验（模拟）。每个模拟目录下包含两个子目录：`character` 和 `grammar`，分别对应于基于字符和基于语法的VAE模型。

- `simulationX/character` 和 `simulationX/grammar`

  - **`results/dummy_file.txt`**:实际使用中存储优化过程中生成的结果数据，如每一步的分子及其属性。

  - **`fpscores.pkl.gz`**: 与顶层`latent_features_and_targets_*`目录中的`fpscores.pkl.gz`相同，包含分子的指纹评分。

  - **`gauss.py`**: 实现高斯过程（Gaussian Process）相关功能，用于贝叶斯优化中的代理模型构建。
 定义和训练高斯过程回归模型，预测潜在空间中的目标函数值。

  - **`random_seed.txt`**: 存储本次模拟使用的随机种子，以确保结果的可复现性。

  - **`run_bo.py`**: 核心的贝叶斯优化脚本，用于执行优化过程。
      1. 初始化贝叶斯优化过程。

      2. 选择下一个采样点（潜在向量）。

      3. 评估目标函数（如SA score）。

      4. 更新代理模型（高斯过程）。

      5. 迭代优化，直到达到预定的步数或终止条件。
- **`sascorer.py`**: 与顶层的`sascorer.py`相同，用于计算分子的合成可行性评分。

  - **`sparse_gp.py`**: 实现稀疏高斯过程（Sparse Gaussian Process），用于处理大规模数据集并提高计算效率,定义稀疏高斯过程模型，训练和预测目标函数值。

  - **`sparse_gp_theano_internal.py`**: 与Theano集成的稀疏高斯过程实现，提供底层计算支持。实现与Theano的接口和计算图，提供高效的稀疏高斯过程计算方法。

##### d. **`get_average_test_RMSE_LL.sh`**

- **说明**: Shell脚本，用于计算和汇总所有模拟实验的平均测试均方根误差（RMSE）和对数似然（LL）。
      1. 遍历所有`simulation*`目录下的`results`文件。
      2. 提取每个模拟的RMSE和LL值。
      3. 计算并输出平均值，以评估优化算法的整体性能。

##### e. **`get_final_results.py`**

- **说明**: Python脚本，用于从所有模拟实验中提取和汇总最终结果，生成综合报告。
       1. 遍历所有`simulation*`目录。
       2. 提取每个模拟的最佳分子及其属性。
       3. 汇总所有模拟的最佳分子，生成最终报告（如`final_results.pkl`）。

### 4. **根目录**

- **`encode_decode_zinc.py`**: 用于使用训练好的VAE模型对SMILES字符串进行编码和解码。
      1. 加载预训练的VAE模型（位于`pretrained`目录）。
      2. 读取示例SMILES字符串。
      3. 编码SMILES字符串为潜在向量。
      4. 解码潜在向量回SMILES字符串。
      5. 显示编码和解码的结果。

- **`make_zinc_dataset_grammar.py`**: 用于生成适用于基于语法的VAE模型的ZINC分子数据集。
      1. 读取`data/250k_rndm_zinc_drugs_clean.smi`中的SMILES字符串。
      2. 进行语法解析和预处理。
      3. 保存处理后的数据集（如`data/eq2_grammar_dataset.h5`）。

- **`make_zinc_dataset_str.py`**: 用于生成适用于基于字符串的VAE模型的ZINC分子数据集。
      1. 读取`data/250k_rndm_zinc_drugs_clean.smi`中的SMILES字符串。
      2. 进行字符串预处理（如字符分割）。
      3. 保存处理后的数据集（如`data/eq2_str_dataset.h5`）。

- **`molecule_vae.py`**: 用于实现分子的VAE模型，包括编码和解码功能。
      1. 定义VAE的编码器和解码器结构。
      2. 提供编码（SMILES → latent vector）和解码（latent vector → SMILES）的函数。
      3. 加载和保存模型权重。

- **`train_zinc.py`**: 用于训练基于语法的ZINC分子VAE模型。
      1. 加载生成的基于语法的数据集。
      2. 初始化基于语法的VAE模型。
      3. 训练模型，并保存训练好的模型权重至`models`目录。

- **`train_zinc_str.py`**: 用于训练基于字符串的ZINC分子VAE模型。
      1. 加载生成的基于字符串的数据集。
      2. 初始化基于字符串的VAE模型。
      3. 训练模型，并保存训练好的模型权重至`models`目录。

- **`zinc_grammar.py`**可能包含与基于语法的ZINC模型相关的辅助功能或配置。 提供配置参数、辅助函数或其他支持代码，具体内容需查看脚本内部。

### 5. **`pretrained` 目录**

 存放预训练的VAE模型文件，供快速编码/解码和实验使用。

- **`zinc_vae_grammar_L56_E100_val.hdf5`**: 基于语法的VAE模型预训练权重文件。直接用于编码和解码SMILES字符串，节省训练时间。
- **`zinc_vae_str_L56_E100_val.hdf5`**: 基于字符串的VAE模型预训练权重文件。直接用于编码和解码SMILES字符串，节省训练时间。

# 三.项目复现指南

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

## 五、总结

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

