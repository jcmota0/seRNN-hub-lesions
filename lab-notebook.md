# Lab notebook

Lab notebook for my BSc dissertation work.
This is a file where I track what I've done, my thoughts on papers I've read, decisions, supervisor meetings, etc.

Contents:

1. Supervisor meetings
2. Work log
3. Decisions
4. Papers
5. Templates for new entries

## 1. Supervisor meetings

### 12 October 2026

Discussed:
- Met the other students who have the same supervisor as me. We talked about our interests and ambitions. Dr Allefeld suggested that people with similar interests work in groups. Since my interests were computational and no one else had a similar direction, he told me to work on my own and start reading papers to get more concrete ideas.

Agreed:
- Read some literature and come back with 2 or 3 more concrete topics

To do:
- [x] Read papers
- [x] Find topics that interest me

### 19 October 2026

Discussed:
- The seRNN paper and some ideas for my topic. My first idea was to change the strength of the wiring cost during training, to mimic how connections differ between children and adults (the brain overgrows connections and then prunes them). It didn't seem strong enough, because something similar already happens in the original paper: the cost keeps pruning the network as it trains.

Agreed:
- Drop the timing idea and look for a question closer to psychology, ideally one that can be compared with patients
- Bring a short proposal for the new question next time

To do:
- [x] Look for a stronger question
- [x] Write a proposal for it (done 21 Oct: hubs and lesions)

### 26 October 2026

Discussed:
- My reading since the last meeting, especially Yang et al. (2019)

Agreed:
- Read Yang et al. properly, since the new question needs networks that learn many tasks
- Get the official code running before planning anything bigger
- Keep a lab notebook of every decision and setting

To do:
- [ ] Run the official seRNN demo
- [ ] Get Yang et al.'s code running on my Mac

## 2. Work log

### 12 October 2026: first question and first replication

Read Achterberg et al. (2023), the paper that introduced spatially embedded recurrent neural networks (seRNNs).
My first question was whether the timing of the wiring cost during training changes the final network.
Before that, I needed to know I could trust my version of the model.
I rebuilt the model in PyTorch and checked the pieces against the authors' published materials: the task, the wiring cost and the network measures all matched.
Then I trained 360 networks, seRNN and L1 (a simple wiring cost that ignores distance).
Compared at the same total connection strength, my seRNNs were more modular, more small-world and had shorter connections than my L1 networks, as the paper reports.
One thing didn't match: my seRNNs were pruned about half as fast as the authors' at the same cost strength, and I couldn't find out why.
By the end of the day I had started to doubt the timing question, and I discussed it with Dr Allefeld on 19 October.

### 13-16 October 2026: literature on brain damage and network robustness

Read about how damage affects brain networks and trained artificial networks.
Whether a wiring cost makes networks more or less robust to damage is an open question, with two competing predictions: the damage spreads out gently, or the hubs make the network fragile.

### 21 October 2026: proposal for the current question

Wrote the proposal for the current project: do networks that learn many tasks under a wiring cost build connector hubs whose damage causes broad, patient-like deficits?

## 3. Decisions

### Count only active units (26 October 2026)

A wiring cost can switch many units off entirely.
All measures use only active units, and lesions are drawn from active units, so that "how much was damaged" is fair across networks.

### Communicability recomputed every step (26 October 2026)

The seRNN cost depends on communicability, which the original code recomputes at every training step.
If I recompute it less often to save time, I will record that as a change and check that it doesn't alter the results.

## 4. Papers

### Achterberg et al. 2023: spatially embedded RNNs

*Nature Machine Intelligence.*

What they did: 100 units on a 3D grid learned a one-step maze task (see a goal, wait, see two options, pick the one closer to the goal).
The wiring cost was connection strength × distance × communicability.

What they found: seRNNs became sparse, modular and small-world.
Goal information gathered in a compact core of units, while choice information stayed spread out.
Networks with a simple cost (L1) showed none of this.

For my project: I use their wiring cost, but with many tasks, more units (256) and lesions.
Note: the paper measures modules on all connections weighted by strength, not on the strongest 10% as the demo does.

### Yang et al. 2019: task representations in multitask networks

*Nature Neuroscience.*

What they did: one network trained on 20 cognitive tasks.

What they found: how much each unit's activity varies within each task groups the units into clusters that match shared abilities, such as holding something in memory.
Switching off a cluster impairs the tasks that use it.
The units have no positions, so the clusters aren't located anywhere.

For my project: I use their 20 tasks and their activity measure, but give units positions and choose lesions by location or wiring.
Watch out: "fractional task variance" is a different measure from task variance.

### Liu et al. 2023: growing brains

*OpenReview.*

What they did: a 400-unit, 20-task network on a 2D grid with a distance-based wiring cost.

What they found: clusters of units doing the same tasks ended up close together in space, unlike in networks without the distance cost.
Only about 100 of the 400 units stayed important.

For my project: a check for my own networks (clusters should end up close together), and a reason to count only active units.
Watch out: they test whether units are close together, not whether they are wired together.

### Mod-Cog (Khona, Chandra et al.)

What it is: Yang's 20 tasks plus harder versions, 82 tasks in all, that need connections between units to solve.

For my project: forces the networks to use connections between units, which the hub measures need.
It's slower to train, so the fallback is the 20 tasks plus some of the harder versions.
To check: the claim that the 20 basic tasks can be solved without connections between units needs a source.

## 5. Templates for new entries

Copy one of these.

Supervisor meeting:

```markdown
### Date

Discussed:
-

Agreed:
-

To do:
- [ ]
```

Work log:

```markdown
### Date: short title

What I did, the settings, the results, and what's next.
```

Decision:

```markdown
### Short title (date)

What I decided and why.
```

Paper:

```markdown
### Authors year: short title

*Journal.*

What they did:

What they found:

For my project:
```
