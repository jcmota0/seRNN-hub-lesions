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

