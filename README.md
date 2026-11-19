# Why is hub damage so devastating?

BSc Psychology with Neuroscience dissertation.

## The question

In stroke patients, where the damage is matters more than how big it is.
Damage to "connector hubs", regions linking many brain networks, impairs many cognitive abilities at once.
Damage of the same size elsewhere impairs far fewer.

Why would a brain be built with such vulnerable hubs?
One idea is that they appear naturally when a brain has to learn many tasks while keeping its wiring cheap.

This project tests that idea in artificial neural networks.
The networks have units placed in 3D space, pay a cost for long connections, and learn 20 cognitive tasks.
After training, units are switched off in one place, like a small stroke, and we measure which tasks break.

## Approach

1. Train spatially embedded recurrent neural networks (seRNNs, Achterberg et al. 2023) on the 20 cognitive tasks of Yang et al. (2019).
2. Compare four versions: no wiring cost, a simple cost, a distance cost, and the full seRNN cost.
3. Find the hubs in each trained network.
4. Damage hubs and non-hubs of the same size, and compare how many kinds of task fail.

## Status

The seRNN paper's main results have been reproduced.
Next: adding the seRNN wiring cost to the 20-task networks.

## Replication of the seRNN paper

Before building on the model, I reproduced the structural results of Achterberg et al. (2023) with the authors' own code.
The full account is in `lab-notebook.md` (19 November 2026).

| File | What it does |
|---|---|
| `replication/authors_code.py` | The authors' task, wiring costs and weight recorder, copied unchanged from their CodeOcean capsule (MIT License) |
| `replication/train_networks.py` | Trains seRNN and L1 networks at the authors' cost strengths, so each has a "twin" in their data |
| `replication/check_topology.m` | Recomputes the authors' stored network measures on a sample of their networks |
| `replication/compare_twins.m` | Compares my networks with their twins, and seRNN with L1 |

To run it:

1. Download the authors' CodeOcean capsule (capsule 2879348) and unzip it to `~/seRNN-capsule`.
2. Set up Python 3.13 and install `requirements.txt`, for example:
   ```
   uv venv --python 3.13 .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   ```
3. Train the networks (about 10 minutes on a MacBook):
   ```
   .venv/bin/python replication/train_networks.py --group seRNN --strengths notebook
   .venv/bin/python replication/train_networks.py --group L1 --strengths notebook
   ```
4. In MATLAB, with the Statistics and Machine Learning Toolbox, run `check_topology.m` and then `compare_twins.m` from the `replication` folder.

`--strengths notebook` uses the seRNN cost strengths written in the authors' notebook (0.000001 to 1) rather than those stored in their data file (0.001 to 0.3); only the notebook's strengths reproduce their networks (see the lab notebook).
