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

### 19 November 2026

Discussed:
- My check of the seRNN demo across many networks: long connections are clearly weaker, but some networks get stuck and whether modules show up depends on how you measure them
- Building on the authors' code in TensorFlow instead of writing my own

Agreed:
- Redo the replication of the seRNN paper step by step so I can explain every part of it
- Use the authors' CodeOcean capsule, and MATLAB through the uni, for their analysis
- After that, add the seRNN wiring cost to Yang et al.'s 20-task networks

To do:
- [ ] Reproduce the seRNN paper's results with the authors' capsule, step by step
- [ ] Agree on self-connections, cost scaling and number of networks

## 2. Work log

### 12 October 2026: first question and first replication

Read Achterberg et al. (2023), the paper that introduced spatially embedded recurrent neural networks (seRNNs).
My first question was whether the timing of the wiring cost during training changes the final network.
Before that, I needed to know I could trust my version of the model.
I rebuilt the model in PyTorch and checked the pieces against the authors' published materials: the task, the wiring cost and the network measures all matched.
Then I trained 360 networks, seRNN and L1 (a simple wiring cost that ignores distance).
Compared at the same total connection strength, my seRNNs were more modular, more small-world and had shorter connections than my L1 networks, as the paper reports.
One thing didn't match: my seRNNs were pruned about half as fast as the authors' at the same cost strength, and I couldn't find out why (solved on 19 November).
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

### 19 November 2026: clean start, the authors' capsule, and my own networks

Set up this repository and notebook from scratch.
Decided to redo the replication of the seRNN paper step by step, so I can explain every part of it myself instead of just running someone else's pipeline.

**The authors' capsule.**
Downloaded the authors' CodeOcean capsule.
It has their training demo, their analysis scripts (in MATLAB) and all their trained networks: 1,001 seRNN and 1,001 L1 networks, plus smaller sets of other variants.
Their official route only trains one demo network and then remakes the paper's figures from their saved networks, so it never checks that newly trained networks behave the same.

**Reproducing their figures.**
Ran their figure script (`figure_replication_code.m`) section by section in MATLAB (R2026b, with the Statistics and Machine Learning Toolbox, which their scripts need for correlations).
To run it outside their Docker setup I only changed where it looks for the data and where it saves the pictures, not the analysis.
Reproduced the paper's main structural figures from their saved networks:

- Accuracy: seRNN and L1 networks learn the task equally well, from about 57% to about 100% over 10 epochs (only networks that end above 90% are included).
- Example seRNN (network 290): connection strength goes down with distance, r = -0.43.
  The strongest connections are self-connections, which cost nothing because their distance is 0, and they are included in that r.
- Across all networks: seRNNs end with a weight-distance correlation of about -0.3 (L1 about 0), modularity of about 0.37 (L1 about 0.16) and small-worldness of about 2.3 (L1 about 2.0).
  seRNNs actually keep a bit more total weight than L1, so the extra modularity is not just from having fewer connections.

**Checking their numbers.**
These figures use numbers the authors computed earlier with another script (`computing_topology.m`), so I checked those numbers myself.
Wrote `replication/check_topology.m`, which takes 10 random seRNN and 10 random L1 networks at the end of training and recomputes the four measures with the authors' own lines of code.
Total weight, weight-distance correlation and modularity matched exactly in all 20 networks.
Small-worldness matched within about 0.5%, which is expected because it is compared against randomly generated networks.

Things I noticed:

- The network number goes with cost strength: low numbers have weak costs and keep lots of weight, high numbers are pruned hard.
- The most heavily pruned seRNNs (total weight about 30) have modularity of about 0.96.
  These aren't real modules: the network has just fallen apart into disconnected pieces.
  So only networks that still do the task should be compared, and in my project I need to check that networks haven't fallen apart before looking for hubs.
- Self-connections are free under the seRNN cost, which is the same issue I need to agree with my supervisor for my own networks.

**Training my own networks.**
Set up a Python environment for the project (`requirements.txt`) and wrote `replication/train_networks.py`.
It runs the authors' training code unchanged (copied into `replication/authors_code.py`) with their settings: 100 units, 10 epochs, 5,120 training problems.
Each of my networks uses the same cost strength as one of theirs, so it has a "twin" in their data.
I trained every 10th one: 101 seRNN and 101 L1 networks, about 2 seconds each.

**A problem with the cost strengths.**
My L1 networks matched their twins, but my seRNNs didn't.
At the strongest costs their seRNNs were pruned down to a total weight of about 30 and failed the task, while mine kept 90-290 and still learned it.
Their saved data says the seRNN cost strengths went from 0.001 to 0.3, but the text in their other notebook (`recording_decoding_demo.ipynb`) says 0.000001 to 1.
When I trained with the notebook's strengths instead, my seRNNs matched their twins (for example 32 vs 30 total weight, both stuck at about 62%).
So it looks like the strengths stored in their data file are mislabelled, and they really trained with 0.000001 to 1.
This probably also explains why my networks on 12 October pruned slower than theirs.
To do: check which range the published paper reports.

**Comparing my networks with theirs.**
With the notebook's strengths, compared all 202 networks with their twins in MATLAB (`replication/compare_twins.m`, using the authors' code for the measures).
Total weight tracks the twin closely (r = 0.97 for L1, 0.94 for seRNN).
Individual pairs differ more for modularity and accuracy, because whether a network gets stuck or falls apart is partly random, and my random seeds are different from theirs.

The main result, using only networks that learned the task (at least 90%), at the end of training:

| | Weight-distance r | Modularity | Small-worldness |
|---|---|---|---|
| Mine, L1 (53 networks) | -0.02 | 0.16 | 1.89 |
| Mine, seRNN (41) | -0.32 | 0.39 | 2.29 |
| Theirs, L1 (46) | -0.01 | 0.14 | 1.96 |
| Theirs, seRNN (31) | -0.32 | 0.31 | 2.41 |

My networks reproduce the paper's structural results: like theirs, seRNNs wire more locally, are more modular and are more small-world than L1 networks.

Next: add statistics like the paper (t-tests and effect sizes), and check the strength range in the published paper.

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
Note: their saved data and their notebook disagree on the seRNN cost strengths (see 19 November).

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
