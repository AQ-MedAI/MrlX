# MrlX-DeepResearch Quick Start Guide

This guide helps you quickly start the Main Agent and Sub Agent training environments with integrated Tool Server.

## Prerequisites

- Two containers configured with 8x H20(141g) GPUs
- At least 2T memory per container
- slime official Docker image installed

## Environment Variables Configuration

Before starting, you need to configure all environment variables. Copy `env.example` to `.env` and fill in the following:

### Required Variables

| Variable | Description | Example | Used By |
|----------|-------------|---------|---------|
| `MAIN_HF_CHECKPOINT` | Main Agent HF format model checkpoint path | `/path/to/Qwen3-30B-A3B-Instruct-2507` | Main Agent |
| `MAIN_REF_LOAD` | Main Agent reference model path (Megatron torch_dist format - **must be converted**) | `/path/to/Qwen3-30B-A3B-Instruct-2507_torch_dist` | Main Agent |
| `SUB_HF_CHECKPOINT` | Sub Agent HF format model checkpoint path | `/path/to/Qwen3-30B-A3B-Instruct-2507` | Sub Agent |
| `SUB_REF_LOAD` | Sub Agent reference model path (Megatron torch_dist format - **must be converted**) | `/path/to/Qwen3-30B-A3B-Instruct-2507_torch_dist` | Sub Agent |
| `SUMMARY_LLM_API_KEY` | Summary LLM API Key | `sk-xxx...` | Main & Sub Agent |
| `SUMMARY_LLM_API_BASE` | Summary LLM API Base URL | `https://openrouter.ai/api/v1` | Main & Sub Agent |
| `SUMMARY_LLM_MODEL` | Summary LLM Model Name | `deepseek/deepseek-chat-v3-0324` | Main & Sub Agent |
| `JUDGE_LLM_API_KEY` | Judge LLM API Key | `sk-xxx...` | Main Agent |
| `JUDGE_LLM_API_BASE` | Judge LLM API Base URL | `https://openrouter.ai/api/v1` | Main Agent |
| `JUDGE_LLM_MODEL` | Judge LLM Model Name | `deepseek/deepseek-chat-v3-0324` | Main Agent |
| `REASONER_LLM_API_KEY` | Reasoner LLM API Key | `sk-xxx...` | Main Agent |
| `REASONER_LLM_API_BASE` | Reasoner LLM API Base URL | `https://openrouter.ai/api/v1` | Main Agent |
| `REASONER_LLM_MODEL` | Reasoner LLM Model Name | `deepseek/deepseek-chat-v3-0324` | Main Agent |
| `SUB_AGENT_IP` | Sub Agent IP address (get from sub container) | `192.168.1.100` | Main Agent |
| `TOOL_SERVER_LLM_API_KEY` | Tool Server LLM API Key | `sk-xxx...` | Tool Server |
| `TOOL_SERVER_LLM_BASE_URL` | Tool Server LLM Base URL | `https://openrouter.ai/api/v1` | Tool Server |
| `TOOL_SERVER_LLM_MODEL` | Tool Server LLM Model | `deepseek/deepseek-chat-v3-0324` | Tool Server |
| `GOOGLE_SEARCH_KEY` | Google Search API Key | `your_key_here` | Tool Server |
| `JINA_API_KEY` | Jina API Key | `your_key_here` | Tool Server |
| `SLIME_DIR` | slime framework directory path | `/path/to/slime` | System |

### Optional Variables

| Variable | Description | Default | Used By |
|----------|-------------|---------|---------|
| `DATABASE_SERVER_HOST` | Database Server Host | `0.0.0.0` | Database Server |
| `DATABASE_SERVER_PORT` | Database Server Port | `18888` | Database Server |
| `DATABASE_SERVER_IP` | Database Server IP (only required when Database Server and Sub Agent are on different containers - uncomment and set if needed) | Uses `SUB_AGENT_IP` | Main & Sub Agent |
| `TOOL_SERVER_PORT` | Tool Server Port | `50001` | Tool Server |
| `MAX_CONCURRENT_REQUESTS` | Max Concurrent Requests | `2000` | Tool Server |
| `VISIT_SEMAPHORE_LIMIT` | Visit Semaphore Limit | `200` | Tool Server |
| `SEARCH_SEMAPHORE_LIMIT` | Search Semaphore Limit | `500` | Tool Server |
| `WEBCONTENT_MAXLENGTH` | Web Content Max Length | `150000` | Tool Server |
| `RETRIEVAL_SERVICE_URL` | Tool Server access URL (only required when Tool Server and Main Agent are on different containers - uncomment and set if needed) | `http://localhost:50001/retrieve` | Main Agent |
| `MAIN_CHECKPOINT_DIR` | Main Agent checkpoint directory | `$SLIME_DIR/../checkpoints/main` | Main Agent |
| `MAIN_PROMPT_DATA` | Main Agent prompt data file | `$SLIME_DIR/examples/agent-co-train/mock_data_main.jsonl` | Main Agent |
| `MAIN_RAY_TEMP_DIR` | Main Agent Ray temp directory | `$SLIME_DIR/../ray_temp/main` | Main Agent |
| `SUB_CHECKPOINT_DIR` | Sub Agent checkpoint directory | `$SLIME_DIR/../checkpoints/sub` | Sub Agent |
| `SUB_PROMPT_DATA` | Sub Agent prompt data file | `$SLIME_DIR/examples/agent-co-train/mock_data_sub.jsonl` | Sub Agent |
| `LOG_DIR` | Log directory | `$SLIME_DIR/logs` | System |

> **Note**:
> - **Database Server**: By default, automatically starts on Sub Agent container (port 18888). Both Main Agent and Sub Agent access it via `SUB_AGENT_IP:18888`. For separate deployment, uncomment `DATABASE_SERVER_IP` in `.env` and set to actual Database Server address.
> - **Tool Server**: By default, automatically starts on Main Agent container (port 50001). Main Agent accesses `http://localhost:50001/retrieve`. For separate deployment, uncomment `RETRIEVAL_SERVICE_URL` in `.env` and set to actual Tool Server address, e.g., `http://<tool_server_ip>:50001/retrieve`.

## Quick Start Steps

### Step 1: Clone slime Framework

Before starting training, you need to clone the slime framework in **both containers**:

```bash
# Choose your desired directory, for example:
cd /path/to/your/workspace

# Clone slime framework
git clone https://github.com/THUDM/slime.git
cd slime
echo "SLIME_DIR=$(pwd)"
```

**Important Notes**:
- Please execute the above commands in **both containers** (Main Agent and Sub Agent)
- **Recommended to use the same path**: Clone to the same path in both containers (e.g., `/path/to/your/workspace/slime`), so you can use the same `.env` file
- If using different paths, you need to prepare separate `.env` files for each container with different `SLIME_DIR` values
- Please record the output path, as you'll need to set it as `SLIME_DIR` in the `.env` file later

### Step 2: Prepare Model Checkpoints

Before starting training, you need to prepare model checkpoints in two formats:

1. **Hugging Face (HF) Format**: Download from Hugging Face or ModelScope
   - Used for `MAIN_HF_CHECKPOINT` and `SUB_HF_CHECKPOINT`
   - Example: `Qwen/Qwen3-30B-A3B-Instruct-2507`

2. **Megatron torch_dist Format**: Convert from HF format using slime conversion tools
   - Used for `MAIN_REF_LOAD` and `SUB_REF_LOAD` (reference model paths)
   - **This conversion is required** for training to work

#### Quick Conversion Example

```bash
cd $SLIME_DIR
source scripts/models/qwen3-30B-A3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/Qwen3-30B-A3B-Instruct-2507 \
    --save /path/to/Qwen3-30B-A3B-Instruct-2507_torch_dist
```

> **Important Note for Qwen3-30B-A3B-Instruct-2507 Model**: When using the Qwen3-30B-A3B-Instruct-2507 model, you need to modify the `--rotary-base` parameter in `slime/scripts/models/qwen3-30B-A3B.sh` to `10000000`.

**For detailed conversion instructions, parameters, and slime run configurations**, please refer to:
-  [slime Quick Start Guide](https://thudm.github.io/slime/zh/get_started/quick_start.html)
-  [slime Usage Documentation](https://thudm.github.io/slime/zh/get_started/usage.html)

After conversion, configure the paths in your `.env` file:
```bash
MAIN_HF_CHECKPOINT=/path/to/Qwen3-30B-A3B-Instruct-2507              # HF format
MAIN_REF_LOAD=/path/to/Qwen3-30B-A3B-Instruct-2507_torch_dist        # Converted torch_dist format

SUB_HF_CHECKPOINT=/path/to/Qwen3-30B-A3B-Instruct-2507               # HF format
SUB_REF_LOAD=/path/to/Qwen3-30B-A3B-Instruct-2507_torch_dist         # Converted torch_dist format
```

### Step 3: Configure Environment Variables

First, get the Sub Agent IP address. In the **Sub Agent container**, run:

```bash
hostname -i
```

Record the IP address output, for example: `192.168.1.100`

Then, in the project root directory:

```bash
# Copy environment variable template
cp env.example .env

# Edit .env file and fill in ALL required variables
vim .env  # or use your preferred editor
```

Make sure to fill in:
- **SLIME_DIR**: slime framework path recorded in Step 1
- Model checkpoint paths (from Step 2)
- All API keys
- Sub Agent IP (obtained above)

### Step 4: Distribute .env File

Copy the filled `.env` file to both containers:

```bash
# To Main Agent container
scp .env main_container:/path/to/project/MrlX-DeepResearch/

# To Sub Agent container
scp .env sub_container:/path/to/project/MrlX-DeepResearch/
```

### Step 5: Start Training (One Command per Container)

> **Important**: Please start Sub Agent first, then Main Agent. Main Agent needs to connect to Sub Agent, and will report errors if Sub Agent is not running.

#### Step 5.1: Start Sub Agent

In **Sub Agent Container**:
```bash
cd MrlX-DeepResearch
bash quick_start_sub.sh
# or
bash quick_start.sh sub
```

This single command will:
- Install slime dependencies
- Initialize Sub Agent environment variables
- Auto-start Database Server (port 18888)
- Auto-start Router service (port 3333)
- Start Sub Agent training

#### Step 5.2: Start Main Agent

In **Main Agent Container**:
```bash
cd MrlX-DeepResearch
bash quick_start_main.sh
# or
bash quick_start.sh main
```

This single command will:
- Install slime dependencies
- Initialize Main Agent environment variables
- Auto-start Tool Server (port 50001)
- Check Database Server connectivity
- Start Main Agent training

> Notes:
> - Each `quick_start` script handles everything from setup to training startup
> - Database Server runs on Sub Agent container by default (port 18888)
> - Tool Server runs on Main Agent container by default (port 50001)
> - Router service runs on Sub Agent container (port 3333)
> - All services start automatically
> - If deploying Database Server or Tool Server on separate containers, configure the corresponding environment variables before starting

Alternative: Manual Service Control
```bash
# Start Database Server separately (e.g., on a separate container)
bash quick_start_database.sh

# Start Tool Server separately (e.g., on a separate container)
bash quick_start_tool.sh

# Or use unified startup script
bash quick_start.sh main  # Also auto-starts all services
bash quick_start.sh sub   # Also auto-starts all services
```

## File Description

### Main Scripts
- `run.sh` - Main training script (accepts `main` or `sub` parameter)
- `env.example` - Environment variable template file (includes all configurations)
- `.env` - Actual environment configuration (create yourself, not committed to git)

### Quick Start Scripts
- `quick_start.sh` - Unified quick start script (with main/sub parameter)
- `quick_start_main.sh` - Main Agent quick start script (auto-starts Tool Server)
- `quick_start_sub.sh` - Sub Agent quick start script (auto-starts Database Server and Router)
- `quick_start_database.sh` - Database Server independent startup script
- `quick_start_tool.sh` - Tool Server independent startup script

### Service Scripts
- `start_router.sh` - Router service startup script (for Sub Agent)
- `tool_server/` - Tool Server module (provides search and retrieval capabilities)
- `../MrlX/db/database_server.py` - Database Server (provides task queue service)

### Environment Initialization
- `init_env/` - Environment variable initialization scripts directory
  - `init_general_env.sh` - Load general environment variables
  - `init_main_agent_env.sh` - Load Main Agent environment variables
  - `init_sub_agent_env.sh` - Load Sub Agent environment variables
  - `init_database_server_env.sh` - Load Database Server environment variables
  - `init_tool_server_env.sh` - Load Tool Server environment variables

## Training Results and Checkpoints

### Checkpoint Storage Locations

Training results and model checkpoints are automatically saved during training:

#### Main Agent Checkpoints
- **Default Location**: `$SLIME_DIR/../checkpoints/main/`
- **Naming Pattern**: `single-main-MMDD/` (where MMDD is the current date)
- **Example**: `single-main-1225/` for December 25th
- **Custom Location**: Set `MAIN_CHECKPOINT_DIR` in `.env` to specify a different path

#### Sub Agent Checkpoints
- **Default Location**: `$SLIME_DIR/../checkpoints/sub/`
- **Naming Pattern**: `single-sub-MMDD/` (where MMDD is the current date)
- **Example**: `single-sub-1225/` for December 25th
- **Custom Location**: Set `SUB_CHECKPOINT_DIR` in `.env` to specify a different path

#### Checkpoint Contents
Each checkpoint directory contains:
- Model weights and optimizer states
- Training configuration files
- Training logs and metrics
- Model metadata

#### Accessing Checkpoints
```bash
# View Main Agent checkpoints
ls -la $SLIME_DIR/../checkpoints/main/

# View Sub Agent checkpoints
ls -la $SLIME_DIR/../checkpoints/sub/

# View specific date's checkpoint
ls -la $SLIME_DIR/../checkpoints/main/single-main-1225/
```

### Log Files
Training logs are saved to:
- **Default Location**: `$SLIME_DIR/logs/`
- **Naming Pattern**: `MMDD/KEY_SUFFIX_agent_YYYY.log`
- **Example**: `1225/slime-co-train-test_main_1430.log`
- **Custom Location**: Set `LOG_DIR` in `.env` to specify a different path

## Important Notes

1. **Startup Order**: Please start Sub Agent before Main Agent to avoid connection errors
2. `.env` file contains sensitive information (API Keys), do not commit to git
3. Network connectivity is required between both containers, Main Agent needs to access Sub Agent
4. First run will install dependencies, which takes time
5. Please verify `SUB_AGENT_IP` is configured correctly, otherwise Main Agent cannot connect
6. **Database Server Deployment**: By default, Database Server automatically starts on the same container as Sub Agent (port 18888). For separate deployment, configure `DATABASE_SERVER_IP`. Check logs at `logs/database_server.log`
7. **Tool Server Deployment**: By default, Tool Server automatically starts on the same container as Main Agent (port 50001). For separate deployment, configure `RETRIEVAL_SERVICE_URL`. Check logs at `tool_server/logs/tool_server.log`
8. Router service automatically starts with Sub Agent (port 3333), check logs at `logs/router.log`
9. Please fill in all required environment variables (especially API keys)

## Troubleshooting

### Issue: .env file not found
**Solution**: Ensure you have copied `env.example` to `.env` and filled in required variables

### Issue: Main Agent cannot connect to Sub Agent
**Solution**:
1. Check if `SUB_AGENT_IP` in `.env` is correct
2. Test connectivity from Main Agent container: `ping $SUB_AGENT_IP`
3. Confirm Sub Agent has started

### Issue: API Key error
**Solution**: Check if the corresponding API Key in `.env` file is correctly filled

### Issue: Tool Server startup failed
**Solution**:
1. Check Tool Server logs: `tail -f tool_server/logs/tool_server.log`
2. Verify Tool Server environment variables are set (especially `GOOGLE_SEARCH_KEY` and `TOOL_SERVER_LLM_API_KEY`)
3. Check if port 50001 is occupied: `lsof -i:50001`
4. Manually restart: `bash quick_start_tool.sh`

### Issue: Port 50001 already in use
**Solution**:
1. Check existing process: `lsof -ti:50001`
2. Kill existing process: `lsof -ti:50001 | xargs kill -9`
3. Restart Tool Server: `bash quick_start_tool.sh`

### Issue: Tool Server not accessible from Main Agent
**Solution**:
1. Verify Tool Server is running: `curl http://localhost:50001/health`
2. If Tool Server and Main Agent are on different containers, check firewall settings and verify `RETRIEVAL_SERVICE_URL` is correctly configured in `.env`
3. Check Tool Server logs: `tail -f tool_server/logs/tool_server.log`

### Issue: Router service startup failed
**Solution**:
1. Check Router logs: `tail -f logs/router.log`
2. Check if port 3333 is occupied: `lsof -i:3333`
3. Kill existing process: `lsof -ti:3333 | xargs kill -9`
4. Manually restart: `bash start_router.sh`

### Issue: Main Agent cannot connect to Sub Agent
**Solution**:
1. Ensure Sub Agent (including Router) is started BEFORE Main Agent
2. Verify Router is running on Sub Agent: `lsof -i:3333` or `netstat -tuln | grep 3333`
3. Check `SUB_AGENT_IP` is correctly set in Main Agent's `.env` file
4. Test connectivity: `ping $SUB_AGENT_IP` from Main Agent container

### Issue: Database Server startup failed
**Solution**:
1. Check Database Server logs: `tail -f logs/database_server.log`
2. Check if port 18888 is occupied: `lsof -i:18888`
3. Kill existing process: `lsof -ti:18888 | xargs kill -9`
4. Manually restart: `bash quick_start_database.sh`

### Issue: Port 18888 already in use
**Solution**:
1. Check existing process: `lsof -ti:18888`
2. Kill existing process: `lsof -ti:18888 | xargs kill -9`
3. Restart Database Server: `bash quick_start_database.sh`

### Issue: Database Server not accessible from Main Agent
**Solution**:
1. Verify Database Server is running: `curl http://$SUB_AGENT_IP:18888/health`
2. Ensure Sub Agent (including Database Server) is started BEFORE Main Agent
3. If Database Server and Sub Agent are on different containers, check firewall settings and verify `DATABASE_SERVER_IP` is correctly configured in `.env`
4. Check Database Server logs: `tail -f logs/database_server.log`
5. Test connectivity from Main Agent container: `ping $SUB_AGENT_IP` or `nc -zv $SUB_AGENT_IP 18888`
