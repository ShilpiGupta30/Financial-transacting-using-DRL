# Python Transaction Scheduling Environment

This package contains a small, deterministic `SchedulingEnv` and a beginner-friendly PyTorch Deep Q-Network (DQN) agent. It does not connect to MySQL or change the C++ engine.

## Workload and actions

The default workload contains three transactions. Each process tracks its transaction ID, burst time, priority, remaining time, waiting time, and completion status. You can pass a custom non-empty list of mappings with `transaction_id`, `burst_time`, and `priority` when creating an environment.

An action is an integer index into the current ready queue. The queue contains unfinished transactions in their original workload order. For example, action `0` runs the first unfinished transaction for one simulated time unit. The action mapping is recalculated after a transaction completes. An invalid action returns an `invalid_action` info flag and a penalty without changing the simulated clock or transaction data.

## Observation and episode behavior

`reset()` returns a tuple of numbers. Its fixed length for a workload is `1 + 6 * transaction_count`: elapsed simulated time, followed for each transaction by its ID, burst time, priority, remaining time, waiting time, and numeric completion flag. This representation can later be converted directly to a PyTorch tensor.

`step(action)` returns `(observation, reward, terminated, truncated, info)`. Every valid execution decreases the selected transaction's remaining time by one and increases each other ready transaction's waiting time by one. The episode terminates when all transactions complete. It truncates only when the configured maximum number of decisions is reached first, so an agent cannot run forever.

The simple reward is:

```text
-0.1 for each valid execution step
-0.01 for each other ready transaction waiting during that step
+1.0 when the selected transaction completes
-1.0 for an invalid action
```

Invalid actions count toward the maximum decision limit but do not change process state or simulated time. Repeated calls after termination or truncation return the same observation and done flags.

## Install and run on Windows

From the project root in PowerShell:

```powershell
py -m venv drl_scheduler\.venv
.\drl_scheduler\.venv\Scripts\python.exe -m pip install --upgrade pip
.\drl_scheduler\.venv\Scripts\python.exe -m pip install -r drl_scheduler\requirements.txt
.\drl_scheduler\.venv\Scripts\python.exe -m unittest discover -s drl_scheduler/tests -v
```

The only third-party dependency is PyTorch. Choose the PyTorch wheel appropriate
for your machine if you need a specific CPU or CUDA build; the requirements file
uses the standard `torch` package.

## Use the DQN agent

The agent reads the observation length and transaction count from an environment.
Its output is one Q-value per possible action position. Before selecting an
action, it reads each transaction's completion flag and masks action positions
that are no longer in the ready queue. Exploration can be enabled with
epsilon-greedy selection; pass `explore=False` to select the highest-valued
eligible action.

```python
from drl_scheduler.agent import DQNAgent
from drl_scheduler.env import SchedulingEnv

env = SchedulingEnv()
agent = DQNAgent.from_environment(env, seed=7)
observation = env.reset()
action = agent.select_action(observation)  # ready-queue index
next_observation, reward, terminated, truncated, info = env.step(action)
agent.remember(observation, action, reward, next_observation, terminated, truncated)
loss = agent.train_step()  # None until a full replay batch is available
```

The replay buffer stores past transitions. Training uses a target network and
the DQN Bellman target; terminal and truncated transitions do not bootstrap.
The training workflow saves the policy weights together with the configuration
needed to rebuild its network. Training metrics include episode reward, total
waiting time, completed transaction count, episode length, and the end flags.
Evaluation loads that saved policy greedily and does not add replay experiences
or perform training updates. By default, model and CSV files go into
`drl_scheduler/models/`.

## Separate variable-size agent

`VariableSizeDQNAgent` is a separate implementation for experimenting with
different transaction counts. `SchedulingEnv.get_process_features()` returns
one five-value row per process in input order: burst time, priority, remaining
time, waiting time, and completion flag. Transaction IDs are excluded. The
shared network has fixed input and output layers and produces one score per row;
the agent masks completed rows and returns an action index into the environment's
current ready queue. Its separate checkpoint was trained for 300 episodes and
does not replace or load the original fixed-size checkpoint.

Train it separately with seeded synthetic workloads whose transaction counts
cycle through the configured range and whose burst times and priorities are
sampled from the configured integer ranges:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.train_variable_size --episodes 300 --seed 7 --min-transactions 1 --max-transactions 8
```

This writes `drl_scheduler/models/variable_dqn_checkpoint.pt` and
`drl_scheduler/models/variable_training_metrics.csv`. The checkpoint records
the feature schema, architecture version, model configuration, workload ranges,
episode count, and seed. The original `dqn_checkpoint.pt` and `train.py` workflow
remain separate. This new checkpoint has not been benchmarked or connected to
the inference adapter or application interfaces.

To run a separate controlled gamma=1.0 experiment with the same workload,
reward, feature, and network defaults, use:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.train_variable_size_gamma1 --episodes 300 --seed 7 --min-transactions 1 --max-transactions 8
```

This writes `drl_scheduler/models/variable_dqn_gamma1_checkpoint.pt` and
`drl_scheduler/models/variable_gamma1_training_metrics.csv`. The checkpoint
stores gamma in its `agent_config`; the experiment uses the existing
`VariableSizeDQNAgent` and can be loaded with `load_variable_checkpoint`.
Neither existing checkpoint nor the original training outputs are overwritten.
The gamma=1.0 model must be evaluated on the same seeded workloads as the
gamma=0.99 checkpoint before drawing conclusions about scheduling performance.

Evaluate the trained variable-size checkpoint separately against FCFS, SJF,
larger-priority-first, and Round Robin on the same seeded workloads:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.evaluate_variable_size --workloads-per-size 5 --seed 7 --round-robin-quantum 1
```

The default workload sizes are 1, 2, 3, 4, 6, and 8 transactions. SJF is
non-preemptive: it chooses the shortest original burst at each job boundary and
runs that job to completion. All policies use one-unit environment steps and
the environment's waiting-time accounting. Turnaround is completion time minus
arrival time (all generated arrivals are zero); makespan is the total elapsed
episode time. The command writes detailed paired results to
`models/variable_evaluation_detailed.csv` and means grouped by workload size and
policy to `models/variable_evaluation_summary.csv`. It does not retrain or
overwrite either checkpoint.

For a larger generalization experiment, run 100 paired workloads per size using
evaluation seeds starting at 10000 (separate from the model's training seed 7):

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.evaluate_variable_size_generalization --workloads-per-size 100 --seed 10000 --round-robin-quantum 1
```

The workload sizes are 1, 2, 3, 4, 6, and 8. The script reports mean and sample
standard deviation of average waiting and turnaround times, plus per-workload
paired DQN-minus-SJF waiting differences and win/tie/loss counts. It writes new
files: `models/variable_generalization_detailed.csv`,
`models/variable_generalization_summary.csv`, `models/variable_generalization_pairs.csv`,
and `models/variable_generalization_paired_summary.csv`; the earlier five-workload
evaluation files remain unchanged. Makespan is reported but is not a useful
discriminator here because every arrival is at time zero and the CPU never
idles.

To compare the gamma=0.99 and gamma=1.0 variable-size checkpoints on the exact
600 workloads stored in `models/variable_generalization_detailed.csv`, run:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.compare_variable_checkpoints
```

The command verifies the checkpoint gamma metadata, evaluates both policies
greedily, and reruns FCFS, non-preemptive SJF, Priority, and Round Robin on the
same workload records. It writes new files named
`models/variable_gamma_comparison_detailed.csv`,
`models/variable_gamma_comparison_summary.csv`,
`models/variable_gamma_comparison_pairs.csv`, and
`models/variable_gamma_comparison_paired_summary.csv`. Paired summaries report
waiting-time wins, ties, and losses for gamma=1.0 versus gamma=0.99 and for each
DQN versus SJF. Existing evaluation and checkpoint files are not overwritten.

To compare all three variable-size checkpoints, including the priority-free
normalized model, on those same 600 workloads, run:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.compare_three_variable_checkpoints
```

The comparison loads the five-feature checkpoints with `load_variable_checkpoint`
and the four-feature priority-free checkpoint with its schema-specific loader.
It reports mean and sample standard deviation for average waiting and turnaround
time, along with paired waiting-time wins/ties/losses against the other DQNs and
SJF. Results go to four new `variable_three_checkpoint_comparison_*.csv` files.
The saved models and existing evaluation CSVs are read-only inputs.

## Schema-aware held-out generalization experiment

Run the corrected seeded evaluation (base seed 20000). It reconstructs training
workloads from checkpoint metadata and compares candidates with all prior
detailed evaluation CSVs:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.evaluate_variable_size_heldout
```

The gamma=0.99 and gamma=1.0 checkpoints use a priority-aware holdout that
excludes previously seen ordered `(burst_time, priority)` profiles. The
priority-free checkpoint gets a stricter model-input-aware cohort that excludes
any previously seen ordered burst-time sequence, regardless of priority. N=1
has no unseen burst values, so strict priority-free generalization is not tested
at that size; N=2 uses only the actual available unseen sequences. The report
and availability CSV disclose exact counts and sample sizes. Output seed fields
include a `seed_kind`; N=1 records a `profile_order_seed`, not a per-workload
random draw. Outputs use new
`variable_heldout_corrected_*` names; prior CSVs, reports, and checkpoints
remain unchanged. Transaction IDs are not used for profile separation because
they are not model inputs.

## Priority-free normalized feature ablation

This separate variable-size model excludes priority and transaction IDs from
its learned inputs. Its four features are normalized burst time, normalized
remaining time, normalized waiting time, and the completion flag. Burst and
remaining time use `max_burst_time` as their scale. Waiting uses
`max(1, (max_transactions - 1) * max_burst_time)`, the maximum service that
other configured transactions can contribute while one waits. All three time
features are therefore within `[0, 1]` for workloads within the configured
limits. It uses the existing environment reward and termination behavior and
gamma=1.0.

Train the ablation separately with:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.train_priority_free_variable_size --episodes 300 --seed 7 --min-transactions 1 --max-transactions 8
```

This writes `models/variable_dqn_priority_free_gamma1_checkpoint.pt` and
`models/variable_priority_free_gamma1_training_metrics.csv`. The checkpoint
records the feature names, normalization scales, architecture, gamma, workload
limits, episode count, and seed. Its four-input architecture is intentionally
incompatible with the existing five-feature checkpoints. Evaluate it on the
same seeded workloads and against SJF before making performance claims.

## Train and evaluate on Windows

From the repository root in PowerShell, using the dedicated DRL environment:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.train train --episodes 300 --seed 7
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.train evaluate --episodes 20 --seed 7
.\drl_scheduler\.venv\Scripts\python.exe -m unittest discover -s drl_scheduler/tests -v
```

Training creates `drl_scheduler/models/dqn_checkpoint.pt` and
`drl_scheduler/models/training_metrics.csv`. Evaluation creates
`drl_scheduler/models/evaluation_metrics.csv`. Override the training artifact
location with `--model-dir`; evaluation accepts `--checkpoint` and `--metrics`
paths. The default training run uses 300 short episodes and CPU execution.

## Greedy inference adapter

`inference_adapter` accepts a JSON object with a `transactions` array. Each
entry requires an integer `transaction_id`, positive integer `burst_time`, and
integer `priority`. It loads `models/dqn_checkpoint.pt`, runs greedy inference
without training, and writes JSON to standard output. The action trace contains
the stable transaction ID selected for each one-unit execution; an ID can
appear multiple times when its burst requires multiple steps.

The current checkpoint supports exactly three transactions and the adapter
rejects other workload sizes. It maps external transaction IDs separately
from the checkpoint's observation IDs, so the action trace always contains the
input IDs. The checkpoint was trained on one synthetic workload; accepting a
matching shape does not guarantee useful decisions for arbitrary burst or
priority distributions.

From the repository root in PowerShell, read a workload file:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.inference_adapter --input .\drl_scheduler\examples\sample_workload.json
```

Or pipe JSON through standard input:

```powershell
Get-Content -Raw .\drl_scheduler\examples\sample_workload.json | .\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.inference_adapter
```

Example input:

```json
{
  "transactions": [
    {"transaction_id": 1001, "burst_time": 3, "priority": 2},
    {"transaction_id": 1002, "burst_time": 2, "priority": 5},
    {"transaction_id": 1003, "burst_time": 1, "priority": 1}
  ]
}
```

The output has an `action_trace` array of transaction IDs, a `steps` count, and
the completed transaction IDs. Use `--checkpoint PATH` to select another
compatible checkpoint.

## Priority-free variable-size greedy inference

The experimental priority-free adapter is separate from the original fixed-size
adapter. It loads
`models/variable_dqn_priority_free_gamma1_checkpoint.pt`, validates the saved
four-feature schema and normalization metadata, and supports 1–8 transactions
with explicit burst times from 1–8. Each input record requires an integer
`transaction_id` and `burst_time`; IDs must be unique. Priority is not required
or used. IDs are retained only to label the emitted trace. This is the recorded
checkpoint's workload range, not a claim of generalization beyond training.

Run it from the repository root in PowerShell with a JSON workload file:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.priority_free_inference_adapter --input .\drl_scheduler\examples\sample_workload.json
```

Or pipe JSON through standard input:

```powershell
@'
{"transactions":[{"transaction_id":42017,"burst_time":3},{"transaction_id":8,"burst_time":2},{"transaction_id":700003,"burst_time":1}]}
'@ | .\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.priority_free_inference_adapter
```

Example workload:

```json
{
  "transactions": [
    {"transaction_id": 42017, "burst_time": 3},
    {"transaction_id": 8, "burst_time": 2},
    {"transaction_id": 700003, "burst_time": 1}
  ]
}
```

The JSON response includes an `action_trace` of stable transaction IDs. Its
`cpp_trace_records` field contains `BURST,id,time` records followed by one
`ACTION,id` record for each unit of work, matching the C++ `dqn_trace` file
format. This adapter only generates the trace; it does not invoke C++ or
integrate with FastAPI or the dashboard.

## Replay DQN inference in the C++ CLI

The standalone integration script invokes the inference adapter, writes its
stable transaction-ID trace in the C++ `BURST`/`ACTION` format, sends the
workload as six-column CSV to `dqn_trace`, validates the C++ JSON, and prints
the final replay result:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe .\drl_scheduler\cpp_dqn_integration.py --input .\drl_scheduler\examples\sample_workload.json
```

The sample contains only scheduling features, so the bridge supplies neutral
CSV defaults (`DEPOSIT`, source account `0`, destination `-1`, amount `0`). A
workload may optionally provide `type`, `source_account_id`,
`destination_account_id`, and `amount` fields. This checkpoint accepts exactly
three transactions with burst times `3, 2, 1` in input order. The script only
bridges local Python and C++ subprocesses; it does not integrate with FastAPI
or the dashboard and does not support arbitrary workload sizes or burst
profiles.

## Compare schedulers

Run the saved DQN checkpoint and deterministic baselines on the same default
workload and reward function:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m drl_scheduler.compare_schedulers
```

The command writes `drl_scheduler/models/scheduler_comparison.csv` with reward,
total waiting time, completed count, and episode length for DQN, FCFS, SJF,
priority scheduling, and Round Robin. The priority baseline runs larger
numeric values first; ties keep the original transaction order. Round Robin
uses a one-unit quantum by default, matching one environment action; change it
with `--round-robin-quantum` if comparing another quantum. The checkpoint can
be selected with `--checkpoint`, and the output path with `--output`.

This comparison uses only the fixed default workload, so identical deterministic
results do not demonstrate performance across other workloads or arrival
patterns. The policies all use the same one-unit actions and environment reward.

## Run tests

From the project root, run:

```powershell
.\drl_scheduler\.venv\Scripts\python.exe -m unittest discover -s drl_scheduler/tests -v
```

Run tests with the DRL virtual environment's interpreter as shown above. The
workflow uses the existing PyTorch dependency and does not require NumPy or any
additional package.

## Current limitations

This is a compact scheduling model with one CPU executing one unit per action. It is not a Gymnasium environment, does not load transactions from MySQL, and does not model resource allocation, synchronization, or financial balances. Training, evaluation, and the baseline comparison apply only to the fixed Python sample workload; they are not connected to the C++ engine or dashboard.
