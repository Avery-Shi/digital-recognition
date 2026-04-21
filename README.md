# 手写数字识别系统

基于 PyTorch 和 Flask 构建的深度学习手写数字识别系统,采用三层 CNN 架构在 MNIST 数据集上实现超过 98% 的准确率。

## 功能特点

- **深度学习模型**: 三层卷积神经网络(32→64→128通道),结合最大池化、Dropout 正则化和全连接层
- **智能图像预处理**: 自动居中、尺寸归一化和 MNIST 标准化处理,确保推理一致性
- **实时推理**: RESTful API 支持 Base64 编码图像输入,返回预测结果和置信度评分
- **一键部署**: 自动化依赖检查、模型训练和服务启动流程

## 项目结构

```
digital-recognition/
├── app.py                  # Flask 后端应用(模型加载、图像预处理、API接口)
├── train_model.py          # 模型训练脚本(数据增强、训练循环、模型保存)
├── requirements.txt        # Python 依赖包
├── digit_recognition_cnn.pth  # 训练后的模型权重              
└── templates/              # 前端页面模板
    ├── index.html         # 默认蓝色主题界面
    ├── index_new.html     # 现代渐变紫色主题界面
    ├── view1.html         # 深色科幻风格界面
    └── view2.html         # 复古绿色终端风格界面
```

## 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.9+
- Flask 2.0+

### 安装步骤

1. **创建虚拟环境(推荐)**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **训练模型**
```bash
python train_model.py
```
系统将自动下载 MNIST 数据集(约 100MB),训练完成后模型保存在 `./data/digit_recognition_cnn.pth`。

4. **启动 Web 服务**
```bash
python app.py
```

5. **访问界面**

浏览器打开以下地址:
- `http://localhost:5000` - 默认界面
- `http://localhost:5000/new` - 现代界面(推荐)
- `http://localhost:5000/view1` - 科幻风格
- `http://localhost:5000/view2` - 复古终端风格

## 技术细节

### 技术栈

- **深度学习框架**: PyTorch 1.9+
- **Web 框架**: Flask 2.0+
- **图像处理**: Pillow, NumPy
- **数据变换**: torchvision.transforms

### 模型架构

```
输入层 (1×28×28)
  ↓
Conv2d(1→32, 3×3) + ReLU + MaxPool(2×2)  [输出: 14×14]
  ↓
Conv2d(32→64, 3×3) + ReLU + MaxPool(2×2)  [输出: 7×7]
  ↓
Conv2d(64→128, 3×3) + ReLU + MaxPool(2×2)  [输出: 3×3]
  ↓
Flatten(1152) → FC(128) + ReLU + Dropout(0.5)
  ↓
FC(10) → LogSoftmax
```

**训练配置**:
- 优化器: Adam (学习率=0.001, 权重衰减=1e-4)
- 损失函数: CrossEntropyLoss
- 批次大小: 64
- 训练轮数: 15
- 数据增强: 随机旋转(±10°)、随机平移(±10%)

### 图像预处理流程

1. Base64 解码 → PIL 灰度图转换
2. 双线性插值缩放到 28×28
3. 阈值分割(>50)提取数字边界框
4. 裁剪并居中放置到 28×28 画布
5. 转换为 Tensor 并归一化到 [0,1]
6. MNIST 标准化: `(x - 0.1307) / 0.3081`

### API 接口

**POST /predict**

请求格式:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

响应格式:
```json
{
  "prediction": 7,
  "confidence": 96.85,
  "probabilities": [0.001, 0.002, ..., 0.968, ...]
}
```

## 使用说明

1. 访问任意界面(推荐 `/new`)
2. 使用鼠标或触摸屏在画布上绘制数字(0-9)
3. 点击"识别"按钮
4. 查看预测结果、置信度和概率分布
5. 点击"清除"重新绘制

## 核心模块

### app.py

- **Net 类**: CNN 网络架构定义(与训练保持一致)
- **preprocess_image()**: 图像预处理函数(居中、归一化、标准化)
- **路由定义**: 多个前端页面入口和预测 API 接口

### train_model.py

- **数据加载**: 自动下载 MNIST 数据集并应用数据增强
- **Net 类**: 三层 CNN 模型定义
- **train()/test()**: 训练和评估函数
- **自动保存**: 根据准确率保存最优模型权重

## 依赖包

```txt
torch>=1.9.0
torchvision>=0.10.0
flask>=2.0.0
numpy>=1.21.0
pillow>=8.3.0
matplotlib>=3.4.0
```

## 性能指标

- **测试准确率**: >98%
- **推理时间**: <50ms/张(CPU)
- **模型大小**: ~5MB
- **输入支持**: 鼠标和触摸屏

## 注意事项

1. 首次运行需下载 MNIST 数据集(约 100MB)
2. 训练时间因硬件而异,通常 5-15 分钟
3. 绘制时尽量居中且填满画布以获得最佳效果
4. 确保 `./data` 目录有写入权限
5. 推荐使用浏览器: Chrome、Firefox、Edge

## 常见问题

**Q: 找不到模型文件?**  
A: 运行 `python train_model.py` 训练模型,或检查 `data/` 目录权限

**Q: 识别准确率低?**  
A: 确保绘制清晰、数字居中且笔画粗细适中

**Q: 端口被占用?**  
A: 修改 `app.py` 第 135 行的端口参数

## 未来扩展

- 添加模型可视化(TensorBoard)
- 支持批量图片上传
- 集成更多数据集(如 EMNIST)
- Docker 容器化部署
- 添加用户认证和历史记录追踪

## 许可证

本项目仅供教育和研究用途。

---

**说明**: 本项目展示了完整的深度学习应用工作流,包括数据准备、模型设计、训练优化和 Web 部署。
