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
- [ ] Read papers
- [ ] Find topics that interest me

## 2. Work log

### 12 October 2026: first question and first replication

Read Achterberg et al. (2023), the paper that introduced spatially embedded recurrent neural networks (seRNNs).
My first question was whether the timing of the wiring cost during training changes the final network.
Before that, I needed to know I could trust my version of the model.
I rebuilt the model in PyTorch and checked the pieces against the authors' published materials: the task, the wiring cost and the network measures all matched.
Then I trained 360 networks, seRNN and L1 (a simple wiring cost that ignores distance).
Compared at the same total connection strength, my seRNNs were more modular, more small-world and had shorter connections than my L1 networks, as the paper reports.
One thing didn't match: my seRNNs were pruned about half as fast as the authors' at the same cost strength, and I couldn't find out why.
By the end of the day I had started to doubt the timing question.

## 3. Decisions

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
