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
- [x] Run the official seRNN demo (27 Oct)
- [x] Get Yang et al.'s code running on my Mac (2 Nov)

### 2 November 2026

Discussed:
- My plan for the hubs and lesions project, and the first run of Yang et al.'s code on my Mac (about 2 hours per network, all 20 tasks learned)
- Some details I had to fix after checking the plan against the code, like what switching off a unit actually means

Agreed:
- Make sure I really understand the seRNN before building on it
- Decide the number of networks later, once I know how fast they train

To do:
- [x] Rebuild the seRNN parts myself to understand them
- [x] Decide which tools to use (13 Nov)
- [x] Check the demo across many networks (17 Nov)

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

### 27 October 2026: the authors' demo

Ran the official seRNN demo unchanged (100 units, one maze task, 10 epochs).
It learned the task perfectly.
But a single network is only one sample, and the demo is a how-to rather than the paper's full analysis.

### 2 November 2026: the 20-task networks

Got Yang et al.'s (2019) 20-task code running on my MacBook.
One network trained in about 2 hours and learned all 20 tasks to at least 90% accuracy, within the range of the authors' own networks.
Also downloaded their 40 trained networks, to test my analysis code against later.
Checked my plan against the official code of both papers and corrected several details, such as how a lesion is defined.

### 2-13 November 2026: rebuilding the seRNN myself

To understand how the model works, I rebuilt its parts step by step: the maze task, the 3D grid of units and the distances between them, the communicability measure, and the wiring cost.
A check on 3 units gave the expected cost and gradient.

### 13 November 2026: tools

Decided to build on the authors' own code in TensorFlow rather than rewrite it (see Decisions).

### 17 November 2026: checking the demo across many networks

Trained about 130 small networks with the demo code, with 5-10 random starting points per setting.

- Some networks get stuck at about 62% accuracy, even with no wiring cost.
  At the demo's cost strength (0.5) most get stuck, so the demo's perfect result on 27 October was partly luck.
  A strength of 0.2 worked for 9 out of 10 networks.
- In seRNNs, long connections are clearly weaker than short ones.
  With the simple L1 cost, distance makes no difference.
- Modules did not show up, because I used the demo's way of measuring them (only the strongest 10% of connections).
  The paper measures them on all connections weighted by strength, and with that method they do show up (see 12 October).

Code written with AI assistance.

## 3. Decisions

### Count only active units (26 October 2026)

A wiring cost can switch many units off entirely.
All measures use only active units, and lesions are drawn from active units, so that "how much was damaged" is fair across networks.

### Communicability recomputed every step (26 October 2026)

The seRNN cost depends on communicability, which the original code recomputes at every training step.
If I recompute it less often to save time, I will record that as a change and check that it doesn't alter the results.

### Build on the authors' code, in TensorFlow (13 November 2026)

Yang et al.'s code already has the 20 tasks, training, the activity measures and lesions, and runs on my MacBook.
I'm not a computer science student, so reusing tested code is safer than rewriting it.
New code: the seRNN wiring cost, hub measures and spatial lesions.

### Number of networks set by measured speed (13 November 2026)

To be fixed once I know how fast networks train on my two Macs and how much time is left.

### Waiting to agree with my supervisor

- [ ] Self-connections switched off in every version, so no version gets them for free.
- [ ] The wiring cost averaged rather than summed over connections, so its strength means the same in networks of different sizes.
- [ ] How many cost strengths and random starting points to train.

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

### Warren et al. 2014

To read.

### Woolgar et al. 2018

To read.

### Béna & Goodman 2025

*Nature Communications.*

To read.
First impression: networks with modules built in by hand don't automatically divide the work between the modules; that mainly happens when the task can be split and resources are tight.
For my project: measure structural and functional modules separately.

### Driscoll, Shenoy & Sussillo 2024

*Nature Neuroscience.*

To read.
First impressions:

- Networks trained on 15 tasks reuse the same building blocks across tasks that need the same computation, such as remembering a direction.
- These building blocks match groups of units only when unit activity can't go negative, which supports using ReLU units.
- Switching off one group impairs only the tasks that use that building block.
- The units have no positions.

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
