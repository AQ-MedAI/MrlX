# MrlX 

Multi-Agent Reinforcement Learning Framework

In MrlX, Agent A and Agent B operate as independent agents, communicating via a message queue that enables cross-agent API calls, abstracts internal logic into external requests, and supports multi-turn interactions, inference result sharing, and collaborative decision-making.

At runtime, Agent A initiates multi-turn dialogue generation, while Agent B also engages in multi-turn responses. The coordination module evaluates both agents’ dialogues, calculates bilateral rewards, and drives iteration through the message queue. Each agent maintains a complete train–infer loop: the Data Buffer manages training samples, the SGLang Router schedules inference tasks, and Megatron executes model training, forming a “Generate → Train → Sync” flywheel mechanism.

Training data flows from the Data Buffer into Megatron, where updated weights are synchronized back to the inference service. This enables efficient knowledge transfer and continuous co-evolution between agents, transcending single-task limitations and allowing multi-agent systems to improve decision-making capabilities in dynamic environments.

## Table of Contents

- [Architecture-Overview](#Architecture-Overview)
- [Use-Cases](#Use-Cases)
  - [MrlX-TakesTwo](#MrlX-TakesTwo)
  - [MrlX-DeepResearch](#MrlX-DeepResearch)
- [Acknowledgements](#Acknowledgements)

## Architecture-Overview

<div align="center">
    <img src="./docs/figs/framework.svg" alt="framework" width="600">
</div>

**Module Description**

- **training (Megatron)**: Responsible for the main training process; reads data from the Data Buffer, trains the model, and synchronizes updated parameters to the Rollout module
- **rollout (SGLang + router)**: Generates new data (including reward calculation and verification) and stores it in the data buffer
- **data buffer**: Serves as a bridge between training and inference, managing prompt initialization, custom data loading, and rollout-generated content
- **custom rollout generation**: Implements custom data generation logic, tailoring multi-turn interaction strategies, output formats
- **message queue**: Transfers multi-turn interaction information between Agent A and Agent B, supports cross-agent API communication, task distribution, and state synchronization, and drives the iterative loop

## Use-Cases

### MrlX-TakesTwo
See [MrlX-TakesTwo](MrlX-TakesTwo/README.md)

### MrlX-DeepResearch
See [MrlX-DeepResearch](MrlX-DeepResearch/README_QUICKSTART.md)


## Acknowledgements

- Special thanks to the following projects & communities: slime, SGLang, Megatron-LM, and others.