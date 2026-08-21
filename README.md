# ASPIRE: Agentic /Skills Discovery for Robotics

[Project Page](https://research.nvidia.com/labs/gear/aspire/) &ensp;|&ensp; [Paper](https://arxiv.org/abs/2607.00272)

<img src="assets/media/covervideo.gif" alt="ASPIRE robot demonstrations" width="100%">

ASPIRE is a new type of continual learning: "training" is skill refinement instead of gradient descent. 

"Trained model" is a repo of sensorimotor skills instead of floating weights. 

“Distributed training” is a panel of agents each practicing a different skill instead of sharded minibatches.

## Quick Start

### Run with a coding agent

To get started with ASPIRE, launch a coding agent such as Codex or Claude Code and enter the following prompt:

```text
Clone the repo: https://github.com/NVlabs/ASPIRE/, Read AGENTS.md and 
run the complete ASPIRE LIBERO-Pro Goal-Swap Quick Start for all ten 
tasks in the libero_goal_swap suite.

Before executing, report the required GPUs, credentials, gated weights, 
services, expected runtime, seed partitions, and output paths. Wait for 
my confirmation before launching. Do not access real-robot code.
```

The canonical procedure for a quick start is [`aspire/sim/.claude/libero/fix-loop/QUICKSTART.md`](aspire/sim/.claude/libero/fix-loop/QUICKSTART.md). The agent must complete preflight and wait for confirmation before installing dependencies, starting services, or launching trials.

**Reference agent environments:** ASPIRE is coding-agent agnostic. Our simulation workflow is packaged for reproduction with Claude Code with Opus 4.6 1M, while the real-robot agent experiments were conducted with Codex. All coding agents can follow the model-neutral instructions in [`AGENTS.md`](AGENTS.md), although orchestration behavior may differ.

### Reproduce full paper results

Name the suite and experiment explicitly. If neither is named, the agent should present this table and stop for selection.

<table>
  <thead>
    <tr>
      <th>Suite</th>
      <th>Experiment</th>
      <th>Runbook</th>
      <th>Video</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LIBERO-Pro</td>
      <td>Fix Loop</td>
      <td><a href="aspire/sim/.claude/libero/fix-loop/INSTRUCTIONS.md"><code>libero/fix-loop/</code></a></td>
      <td rowspan="3"><video src="https://github.com/user-attachments/assets/15c0b425-9fe9-4313-8a5b-8c7da54c24b7" width="240" controls></video></td>
    </tr>
    <tr>
      <td>Robosuite</td>
      <td>Fix Loop</td>
      <td><a href="aspire/sim/.claude/robosuite/fix-loop/INSTRUCTIONS.md"><code>robosuite/fix-loop/</code></a></td>
    </tr>
    <tr>
      <td>BEHAVIOR-1K</td>
      <td>Fix Loop</td>
      <td><a href="aspire/sim/.claude/behavior/fix-loop/INSTRUCTIONS.md"><code>behavior/fix-loop/</code></a></td>
    </tr>
    <tr>
      <td>LIBERO-Pro</td>
      <td>Evolutionary Search</td>
      <td><a href="aspire/sim/.claude/libero/evosearch/INSTRUCTIONS.md"><code>libero/evosearch/</code></a></td>
      <td rowspan="2"><video src="https://github.com/user-attachments/assets/edd2e5e1-b7d6-408a-bb62-23729df14db2" width="240" controls></video></td>
    </tr>
    <tr>
      <td>Robosuite</td>
      <td>Evolutionary Search</td>
      <td><a href="aspire/sim/.claude/robosuite/evosearch/INSTRUCTIONS.md"><code>robosuite/evosearch/</code></a></td>
    </tr>
    <tr>
      <td>LIBERO</td>
      <td>Zero-Shot Transfer</td>
      <td><a href="aspire/sim/.claude/libero/zeroshot-transfer/INSTRUCTIONS.md"><code>libero/zeroshot-transfer/</code></a></td>
      <td rowspan="4"><video src="https://github.com/user-attachments/assets/8676fa10-6719-477f-9d60-a4ad241a3de3" width="240" controls></video></td>
    </tr>
    <tr>
      <td>LIBERO-Long-Pro</td>
      <td>Library-Size Scaling</td>
      <td><a href="aspire/sim/.claude/libero/library-size-scaling/INSTRUCTIONS.md"><code>libero/library-size-scaling/</code></a></td>
    </tr>
    <tr>
      <td>LIBERO-Long-Pro</td>
      <td>Inference-Time Scaling</td>
      <td><a href="aspire/sim/.claude/libero/inference-time-scaling/INSTRUCTIONS.md"><code>libero/inference-time-scaling/</code></a></td>
    </tr>
    <tr>
      <td>Robosuite</td>
      <td>Training Law</td>
      <td><a href="aspire/sim/.claude/robosuite/training-law/INSTRUCTIONS.md"><code>robosuite/training-law/</code></a></td>
    </tr>
    <tr>
      <td>YAM Bimanual</td>
      <td>Sim-to-Real</td>
      <td><a href="aspire/real/README.md"><code>aspire/real/</code></a></td>
      <td><video src="https://github.com/user-attachments/assets/91b38cd3-3a38-4f56-9758-f986d50c5956" width="240" controls></video></td>
    </tr>
  </tbody>
</table>


Before a paper-scale launch, the agent must report the selected tasks, seed schedule, expected trial count and runtime, GPU and credential requirements, services, and output paths, then wait for explicit confirmation.

> [!WARNING]
> ASPIRE executes language-model-generated Python with full import access. Trial processes and watchdogs are not a security sandbox. Run generated code on an isolated host without credentials or sensitive mounts, restrict network access, and never grant a simulation agent access to physical hardware. Real-robot work requires the controls in [`aspire/real/AGENTS.md`](aspire/real/AGENTS.md) and separate operator authorization.

## Contribution Guidelines

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Third-party contributions must include a Developer Certificate of Origin sign-off.

## License

ASPIRE material owned by NVIDIA or contributed under the project license is
available under the [Apache License 2.0](LICENSE). Third-party materials retain
their original terms. See [NOTICE](NOTICE) and the central
[licensing and compliance index](LICENSES/README.md) before using or
redistributing the full stack.

## Citation

If you find ASPIRE useful in your research, please cite:

```bibtex
@article{lu2026aspire,
  title   = {ASPIRE: Agentic /Skills Discovery for Robotics},
  author  = {Runyu Lu and Yubo Wu and Ethan Kou and Letian Fu and Wenli Xiao and
             Ajay Mandlekar and Yinzhen Xu and Guanya Shi and Ken Goldberg and
             Ang Chen and Mosharaf Chowdhury and Yuke Zhu and Linxi "Jim" Fan and Guanzhi Wang},
  year    = {2026},
  journal = {arXiv preprint arXiv:2607.00272},
  url     = {https://arxiv.org/abs/2607.00272}
}
```

ASPIRE was developed by researchers from NVIDIA, the University of Michigan, the University of Illinois Urbana-Champaign, UC Berkeley, and Carnegie Mellon University.
