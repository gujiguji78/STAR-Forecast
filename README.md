STAR-Forecast/
├── 📁 agents/                      # 多智能体系系统
│   ├── autogen_system.py          # AutoGen智能体系统
│   ├── debate_system.py           # 辩论式智能体
│   └── memory_bank.py             # 记忆银行
├── 📁 server/                      # 服务端
│   ├── agent_service.py           # 智能体服务
│   ├── autogen_service.py         # AutoGen服务
│   ├── model_service.py           # 模型服务
│   ├── run_server.py              # 服务器启动
│   └── schemas.py                 # 数据模型
├── 📁 models/                      # 深度学习模型
│   ├── istr.py                    # ISTR前置网络
│   ├── predictor.py               # 预测头网络
│   └── ensemble.py                # 集成预测
├── 📁 training/                    # 训练模块
│   ├── lightning_client.py        # Agent Lightning客户端
│   └── callbacks.py               # 训练回调
├── 📁 data/                        # 数据处理
├── 📁 experiments/                 # 实验管理
├── 📁 results/                     # 训练结果
├── 📁 client/                      # 客户端
├── 📁 deployment/                  # 部署配置
├── 📁 tests/                       # 测试
├── main.py                         # 主程序
└── requirements.txt               # 依赖项
