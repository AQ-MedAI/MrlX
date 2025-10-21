# MrlX-DeepResearch 快速启动指南

本指南帮助您快速启动集成了 Tool Server 的 Main Agent 和 Sub Agent 训练环境。

## 前置要求

- 两个容器，各配置 8x H20(141g) GPU
- 每个容器至少 2T 内存
- 已安装 slime 官方 Docker 镜像

## 环境变量配置

开始之前，您需要配置所有环境变量。复制 `env.example` 到 `.env` 并填写以下内容：

### 必填变量

| 变量名 | 描述 | 示例 | 使用方 |
|--------|------|------|--------|
| `MAIN_HF_CHECKPOINT` | Main Agent HF 格式模型检查点路径 | `/path/to/Qwen3-30B-A3B-Instruct-2507` | Main Agent |
| `MAIN_REF_LOAD` | Main Agent 参考模型路径（Megatron torch_dist 格式 - **必须转换**） | `/path/to/Qwen3-30B-A3B-Instruct-2507_torch_dist` | Main Agent |
| `SUB_HF_CHECKPOINT` | Sub Agent HF 格式模型检查点路径 | `/path/to/Qwen3-30B-A3B-Instruct-2507` | Sub Agent |
| `SUB_REF_LOAD` | Sub Agent 参考模型路径（Megatron torch_dist 格式 - **必须转换**） | `/path/to/Qwen3-30B-A3B-Instruct-2507_torch_dist` | Sub Agent |
| `SUMMARY_LLM_API_KEY` | Summary LLM API Key | `sk-xxx...` | Main & Sub Agent |
| `SUMMARY_LLM_API_BASE` | Summary LLM API Base URL | `https://openrouter.ai/api/v1` | Main & Sub Agent |
| `SUMMARY_LLM_MODEL` | Summary LLM 模型名称 | `deepseek/deepseek-chat-v3-0324` | Main & Sub Agent |
| `JUDGE_LLM_API_KEY` | Judge LLM API Key | `sk-xxx...` | Main Agent |
| `JUDGE_LLM_API_BASE` | Judge LLM API Base URL | `https://openrouter.ai/api/v1` | Main Agent |
| `JUDGE_LLM_MODEL` | Judge LLM 模型名称 | `deepseek/deepseek-chat-v3-0324` | Main Agent |
| `REASONER_LLM_API_KEY` | Reasoner LLM API Key | `sk-xxx...` | Main Agent |
| `REASONER_LLM_API_BASE` | Reasoner LLM API Base URL | `https://openrouter.ai/api/v1` | Main Agent |
| `REASONER_LLM_MODEL` | Reasoner LLM 模型名称 | `deepseek/deepseek-chat-v3-0324` | Main Agent |
| `SUB_AGENT_IP` | Sub Agent IP 地址（从 sub 容器获取） | `192.168.1.100` | Main Agent |
| `TOOL_SERVER_LLM_API_KEY` | Tool Server LLM API Key | `sk-xxx...` | Tool Server |
| `TOOL_SERVER_LLM_BASE_URL` | Tool Server LLM Base URL | `https://openrouter.ai/api/v1` | Tool Server |
| `TOOL_SERVER_LLM_MODEL` | Tool Server LLM 模型 | `deepseek/deepseek-chat-v3-0324` | Tool Server |
| `GOOGLE_SEARCH_KEY` | Google Search API Key | `your_key_here` | Tool Server |
| `JINA_API_KEY` | Jina API Key | `your_key_here` | Tool Server |
| `SLIME_DIR` | slime 框架目录路径 | `/path/to/slime` | 系统 |

### 可选变量

| 变量名 | 描述 | 默认值 | 使用方 |
|--------|------|--------|--------|
| `DATABASE_SERVER_HOST` | Database Server 主机地址 | `0.0.0.0` | Database Server |
| `DATABASE_SERVER_PORT` | Database Server 端口 | `18888` | Database Server |
| `DATABASE_SERVER_IP` | Database Server IP（仅当 Database Server 和 Sub Agent 在不同容器时需要取消注释并设置） | 使用 `SUB_AGENT_IP` | Main & Sub Agent |
| `TOOL_SERVER_PORT` | Tool Server 端口 | `50001` | Tool Server |
| `MAX_CONCURRENT_REQUESTS` | 最大并发请求数 | `2000` | Tool Server |
| `VISIT_SEMAPHORE_LIMIT` | 访问信号量限制 | `200` | Tool Server |
| `SEARCH_SEMAPHORE_LIMIT` | 搜索信号量限制 | `500` | Tool Server |
| `WEBCONTENT_MAXLENGTH` | Web 内容最大长度 | `150000` | Tool Server |
| `RETRIEVAL_SERVICE_URL` | Tool Server 访问地址（仅当 Tool Server 和 Main Agent 在不同容器时需要取消注释并设置） | `http://localhost:50001/retrieve` | Main Agent |
| `MAIN_CHECKPOINT_DIR` | Main Agent 检查点目录 | `$SLIME_DIR/../checkpoints/main` | Main Agent |
| `MAIN_PROMPT_DATA` | Main Agent 提示数据文件 | `$SLIME_DIR/examples/agent-co-train/mock_data_main.jsonl` | Main Agent |
| `MAIN_RAY_TEMP_DIR` | Main Agent Ray 临时目录 | `$SLIME_DIR/../ray_temp/main` | Main Agent |
| `SUB_CHECKPOINT_DIR` | Sub Agent 检查点目录 | `$SLIME_DIR/../checkpoints/sub` | Sub Agent |
| `SUB_PROMPT_DATA` | Sub Agent 提示数据文件 | `$SLIME_DIR/examples/agent-co-train/mock_data_sub.jsonl` | Sub Agent |
| `LOG_DIR` | 日志目录 | `$SLIME_DIR/logs` | 系统 |

> **注意**：
> - **Database Server**：默认在 Sub Agent 容器上自动启动（端口 18888），Main Agent 和 Sub Agent 都会通过 `SUB_AGENT_IP:18888` 访问。如需独立部署，请在 `.env` 中取消注释 `DATABASE_SERVER_IP` 并设置为 Database Server 的实际地址。
> - **Tool Server**：默认在 Main Agent 容器上自动启动（端口 50001），Main Agent 会访问 `http://localhost:50001/retrieve`。如需独立部署，请在 `.env` 中取消注释 `RETRIEVAL_SERVICE_URL` 并设置为 Tool Server 的实际地址，例如：`http://<tool_server_ip>:50001/retrieve`。

## 快速启动步骤

### 步骤 1：Clone slime 框架

在开始训练之前，您需要在**两个容器**中都 clone slime 框架：

```bash
# 选择您想要的目录，例如：
cd /path/to/your/workspace

# Clone slime 框架
git clone https://github.com/THUDM/slime.git
cd slime
echo "SLIME_DIR=$(pwd)"
```

**重要说明**：
- 请在**两个容器**（Main Agent 和 Sub Agent）中都执行上述命令
- **建议使用相同路径**：两个容器都 clone 到相同的路径（如 `/path/to/your/workspace/slime`），这样可以使用同样的 `.env` 文件
- 如果使用不同路径，需要为两个容器分别准备不同的 `.env` 文件，分别设置 `SLIME_DIR`
- 请记录输出的路径，稍后需要在 `.env` 文件中设置为 `SLIME_DIR` 的值

### 步骤 2：准备模型检查点

在开始训练之前，您需要准备两种格式的模型检查点：

1. **Hugging Face (HF) 格式**：从 Hugging Face 或 ModelScope 下载
   - 用于 `MAIN_HF_CHECKPOINT` 和 `SUB_HF_CHECKPOINT`
   - 示例：`Qwen/Qwen3-30B-A3B-Instruct-2507`

2. **Megatron torch_dist 格式**：使用 slime 转换工具从 HF 格式转换
   - 用于 `MAIN_REF_LOAD` 和 `SUB_REF_LOAD`（参考模型路径）
   - **此转换是必需的**，训练才能正常工作

#### 快速转换示例

```bash
cd $SLIME_DIR
source scripts/models/qwen3-30B-A3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/Qwen3-30B-A3B-Instruct-2507 \
    --save /path/to/Qwen3-30B-A3B-Instruct-2507_torch_dist
```
> **Qwen3-30B-A3B-Instruct-2507 模型重要提醒**：使用 Qwen3-30B-A3B-Instruct-2507 模型时，需要将 `slime/scripts/models/qwen3-30B-A3B.sh` 中的 `--rotary-base` 参数修改为 `10000000`。

**详细的转换说明、参数配置以及 slime run 配置**，请参考：
-  [slime 快速入门指南](https://thudm.github.io/slime/zh/get_started/quick_start.html)
-  [slime 使用文档](https://thudm.github.io/slime/zh/get_started/usage.html)

转换完成后，在 `.env` 文件中配置路径：
```bash
MAIN_HF_CHECKPOINT=/path/to/Qwen3-30B-A3B-Instruct-2507              # HF 格式
MAIN_REF_LOAD=/path/to/Qwen3-30B-A3B-Instruct-2507_torch_dist        # 转换后的 torch_dist 格式

SUB_HF_CHECKPOINT=/path/to/Qwen3-30B-A3B-Instruct-2507               # HF 格式
SUB_REF_LOAD=/path/to/Qwen3-30B-A3B-Instruct-2507_torch_dist         # 转换后的 torch_dist 格式
```

### 步骤 3：配置环境变量

首先，获取 Sub Agent IP 地址。在 **Sub Agent 容器** 中执行：

```bash
hostname -i
```

记录输出的 IP 地址，例如：`192.168.1.100`

然后，在项目根目录：

```bash
# 复制环境变量模板
cp env.example .env

# 编辑 .env 文件并填写上述所有必填变量
vim .env  # 或使用您喜欢的编辑器
```

确保填写以下内容：
- **SLIME_DIR**：步骤 1 中记录的 slime 框架路径
- 模型检查点路径（来自步骤 2）
- 所有 API keys
- Sub Agent IP（上面获取的）

### 步骤 4：分发 .env 文件

将填好的 `.env` 文件复制到两个容器：

```bash
# 复制到 Main Agent 容器
scp .env main_container:/path/to/project/MrlX-DeepResearch/

# 复制到 Sub Agent 容器
scp .env sub_container:/path/to/project/MrlX-DeepResearch/
```

### 步骤 5：启动训练（每个容器一条命令）

> **重要**：请先启动 Sub Agent，再启动 Main Agent。Main Agent 需要连接 Sub Agent，如果 Sub Agent 未运行会报错。

#### 步骤 5.1：启动 Sub Agent

在 **Sub Agent 容器**中：
```bash
cd MrlX-DeepResearch
bash quick_start_sub.sh
# 或
bash quick_start.sh sub
```

这一条命令会：
- 安装 slime 依赖
- 初始化 Sub Agent 环境变量
- 自动启动 Database Server（端口 18888）
- 自动启动 Router 服务（端口 3333）
- 启动 Sub Agent 训练

#### 步骤 5.2：启动 Main Agent

在 **Main Agent 容器**中：
```bash
cd MrlX-DeepResearch
bash quick_start_main.sh
# 或
bash quick_start.sh main
```

这一条命令会：
- 安装 slime 依赖
- 初始化 Main Agent 环境变量
- 自动启动 Tool Server（端口 50001）
- 检查 Database Server 连通性
- 启动 Main Agent 训练

> 提示：
> - 每个 `quick_start` 脚本会处理从环境配置到训练启动的所有步骤
> - Database Server 默认运行在 Sub Agent 容器上（端口 18888）
> - Tool Server 默认运行在 Main Agent 容器上（端口 50001）
> - Router 服务运行在 Sub Agent 容器上（端口 3333）
> - 所有服务都会自动启动
> - 如需将 Database Server 或 Tool Server 部署在独立容器上，请在启动前配置相应的环境变量

可选：手动控制服务
```bash
# 单独启动 Database Server（例如部署在独立容器上）
bash quick_start_database.sh

# 单独启动 Tool Server（例如部署在独立容器上）
bash quick_start_tool.sh

# 或使用统一启动脚本
bash quick_start.sh main  # 也会自动启动所有服务
bash quick_start.sh sub   # 也会自动启动所有服务
```

## 文件说明

### 主要脚本
- `run.sh` - 主训练脚本（接受 `main` 或 `sub` 参数）
- `env.example` - 环境变量模板文件（包含所有配置）
- `.env` - 实际环境配置（自行创建，不提交到 git）

### 快速启动脚本
- `quick_start.sh` - 统一快速启动脚本（带 main/sub 参数）
- `quick_start_main.sh` - Main Agent 快速启动脚本（自动启动 Tool Server）
- `quick_start_sub.sh` - Sub Agent 快速启动脚本（自动启动 Database Server 和 Router）
- `quick_start_database.sh` - Database Server 独立启动脚本
- `quick_start_tool.sh` - Tool Server 独立启动脚本

### 服务脚本
- `start_router.sh` - Router 服务启动脚本（用于 Sub Agent）
- `tool_server/` - Tool Server 模块（提供搜索和检索能力）
- `../MrlX/db/database_server.py` - Database Server（提供任务队列服务）

### 环境初始化
- `init_env/` - 环境变量初始化脚本目录
  - `init_general_env.sh` - 加载通用环境变量
  - `init_main_agent_env.sh` - 加载 Main Agent 环境变量
  - `init_sub_agent_env.sh` - 加载 Sub Agent 环境变量
  - `init_database_server_env.sh` - 加载 Database Server 环境变量
  - `init_tool_server_env.sh` - 加载 Tool Server 环境变量

## 训练结果和检查点

### 检查点存储位置

训练结果和模型检查点在训练过程中会自动保存：

#### Main Agent 检查点
- **默认位置**：`$SLIME_DIR/../checkpoints/main/`
- **命名模式**：`single-main-MMDD/`（其中 MMDD 是当前日期）
- **示例**：`single-main-1225/` 表示 12月25日
- **自定义位置**：在 `.env` 中设置 `MAIN_CHECKPOINT_DIR` 来指定不同路径

#### Sub Agent 检查点
- **默认位置**：`$SLIME_DIR/../checkpoints/sub/`
- **命名模式**：`single-sub-MMDD/`（其中 MMDD 是当前日期）
- **示例**：`single-sub-1225/` 表示 12月25日
- **自定义位置**：在 `.env` 中设置 `SUB_CHECKPOINT_DIR` 来指定不同路径

#### 检查点内容
每个检查点目录包含：
- 模型权重和优化器状态
- 训练配置文件
- 训练日志和指标
- 模型元数据

#### 访问检查点
```bash
# 查看 Main Agent 检查点
ls -la $SLIME_DIR/../checkpoints/main/

# 查看 Sub Agent 检查点
ls -la $SLIME_DIR/../checkpoints/sub/

# 查看特定日期的检查点
ls -la $SLIME_DIR/../checkpoints/main/single-main-1225/
```

### 日志文件
训练日志保存到：
- **默认位置**：`$SLIME_DIR/logs/`
- **命名模式**：`MMDD/KEY_SUFFIX_agent_YYYY.log`
- **示例**：`1225/slime-co-train-test_main_1430.log`
- **自定义位置**：在 `.env` 中设置 `LOG_DIR` 来指定不同路径

## 注意事项

1. **启动顺序**：请先启动 Sub Agent，再启动 Main Agent，避免连接错误
2. `.env` 文件包含敏感信息（API Keys），请勿提交到 git
3. 两个容器之间需要网络互通，Main Agent 需要访问 Sub Agent
4. 首次运行会安装依赖，需要一定时间
5. 请确认 `SUB_AGENT_IP` 配置正确，否则 Main Agent 无法连接
6. **Database Server 部署**：默认情况下，Database Server 会随 Sub Agent 自动启动在同一容器上（端口 18888）。如需独立部署，请配置 `DATABASE_SERVER_IP`。日志位于 `logs/database_server.log`
7. **Tool Server 部署**：默认情况下，Tool Server 会随 Main Agent 自动启动在同一容器上（端口 50001）。如需独立部署，请配置 `RETRIEVAL_SERVICE_URL`。日志位于 `tool_server/logs/tool_server.log`
8. Router 服务会随 Sub Agent 自动启动（端口 3333），日志位于 `logs/router.log`
9. 请填写所有必需的环境变量（特别是 API keys）

## 故障排查

### 问题：找不到 .env 文件
**解决方案**：确保已将 `env.example` 复制为 `.env` 并填写必需变量

### 问题：Main Agent 无法连接到 Sub Agent
**解决方案**：
1. 检查 `.env` 中的 `SUB_AGENT_IP` 是否正确
2. 从 Main Agent 容器测试连通性：`ping $SUB_AGENT_IP`
3. 确认 Sub Agent 已启动

### 问题：API Key 错误
**解决方案**：检查 `.env` 文件中对应的 API Key 是否正确填写

### 问题：Tool Server 启动失败
**解决方案**：
1. 查看 Tool Server 日志：`tail -f tool_server/logs/tool_server.log`
2. 验证 Tool Server 环境变量已设置（特别是 `GOOGLE_SEARCH_KEY` 和 `TOOL_SERVER_LLM_API_KEY`）
3. 检查端口 50001 是否被占用：`lsof -i:50001`
4. 手动重启：`bash quick_start_tool.sh`

### 问题：端口 50001 已被占用
**解决方案**：
1. 查看占用端口的进程：`lsof -ti:50001`
2. 终止占用进程：`lsof -ti:50001 | xargs kill -9`
3. 重启 Tool Server：`bash quick_start_tool.sh`

### 问题：Main Agent 无法访问 Tool Server
**解决方案**：
1. 验证 Tool Server 正在运行：`curl http://localhost:50001/health`
2. 如果 Tool Server 和 Main Agent 在不同容器上，检查防火墙设置并确认已在 `.env` 中正确配置 `RETRIEVAL_SERVICE_URL`
3. 检查 Tool Server 日志：`tail -f tool_server/logs/tool_server.log`

### 问题：Router 服务启动失败
**解决方案**：
1. 查看 Router 日志：`tail -f logs/router.log`
2. 检查端口 3333 是否被占用：`lsof -i:3333`
3. 终止占用进程：`lsof -ti:3333 | xargs kill -9`
4. 手动重启：`bash start_router.sh`

### 问题：Main Agent 无法连接到 Sub Agent
**解决方案**：
1. 确保在启动 Main Agent 之前已启动 Sub Agent（包括 Router）
2. 验证 Router 在 Sub Agent 上运行：`lsof -i:3333` 或 `netstat -tuln | grep 3333`
3. 检查 Main Agent 的 `.env` 文件中 `SUB_AGENT_IP` 是否正确设置
4. 测试连通性：从 Main Agent 容器执行 `ping $SUB_AGENT_IP`

### 问题：Database Server 启动失败
**解决方案**：
1. 查看 Database Server 日志：`tail -f logs/database_server.log`
2. 检查端口 18888 是否被占用：`lsof -i:18888`
3. 终止占用进程：`lsof -ti:18888 | xargs kill -9`
4. 手动重启：`bash quick_start_database.sh`

### 问题：端口 18888 已被占用
**解决方案**：
1. 查看占用端口的进程：`lsof -ti:18888`
2. 终止占用进程：`lsof -ti:18888 | xargs kill -9`
3. 重启 Database Server：`bash quick_start_database.sh`

### 问题：Main Agent 无法访问 Database Server
**解决方案**：
1. 验证 Database Server 正在运行：`curl http://$SUB_AGENT_IP:18888/health`
2. 确保在启动 Main Agent 之前已启动 Sub Agent（包括 Database Server）
3. 如果 Database Server 和 Sub Agent 在不同容器上，检查防火墙设置并确认已在 `.env` 中正确配置 `DATABASE_SERVER_IP`
4. 检查 Database Server 日志：`tail -f logs/database_server.log`
5. 测试连通性：从 Main Agent 容器执行 `ping $SUB_AGENT_IP` 或 `nc -zv $SUB_AGENT_IP 18888`
