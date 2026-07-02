# 诊断数据分析数据挖掘展示

这是一个面向工业故障诊断部门数据分析/数据挖掘岗位的 Streamlit 展示项目。

当前仓库使用 `price_input_foreign.csv` 中的国外逐小时电价时序作为可替换样例数据，将其抽象为工业状态信号，展示从数据清洗、时序诊断、频域分析、异常识别、工况聚类到预测基线的完整流程。后续可替换为振动、电流、温度、牵引系统或轨旁监测数据。

## 技术定位

- 核心技术栈：信号处理（傅里叶变换、多尺度残差，可扩展小波分析/包络解调）+ 传统机器学习（SVM、随机森林、XGBoost）+ Python/Matlab。
- 技术特点：深度绑定工业故障诊断与信号分析，以传统算法和机理结合为主，深度学习作为后续增强项。
- 当前实现：在不强依赖 `scikit-learn` 与 `xgboost` 的情况下，使用 `numpy/pandas/plotly/streamlit` 完成可运行展示。

## 页面模块

- 诊断总览：有效样本、极值、异常小时、负值小时、时序曲线和异常点标注。
- 信号分析：FFT 主周期识别、多尺度残差能量、小时-月份工况热力图、日级工况聚类。
- 模型评估：基于滞后项、周期编码和滚动统计的 Ridge 时序预测基线。
- 项目展示：岗位匹配说明、GitHub 与 Streamlit 展示入口、数据质量审计。

## 本地运行

```powershell
python -m streamlit run app.py
```

## 链接配置

后续拿到真实 GitHub 仓库链接后，有两种替换方式：

1. 修改 `app.py` 顶部常量：

```python
GITHUB_URL = "https://github.com/ZHXwudi/yuce"
STREAMLIT_URL = "https://your-app.streamlit.app"
```

2. 或在部署平台设置环境变量：

```text
PROJECT_GITHUB_URL=https://github.com/ZHXwudi/yuce
PROJECT_STREAMLIT_URL=https://your-app.streamlit.app
```

## 数据替换建议

若后续接入真实设备数据，建议保留以下字段映射：

- `timestamp`：采样时间。
- `signal` 或业务字段：振动、电流、温度、转速、压力、牵引系统状态量等。
- 可选标签：故障类型、检修记录、设备编号、线路/工况信息。

在真实诊断项目中，可以进一步加入：

- 包络谱、阶次分析、小波包能量、峭度/偏度/RMS/峰峰值等信号特征。
- SVM、随机森林、XGBoost 的监督分类或回归模型。
- 基于检修记录的故障标签闭环和特征重要性解释。
