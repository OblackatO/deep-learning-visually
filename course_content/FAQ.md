# FAQ

<!-- SPDX-FileCopyrightText: 2026 Pedro Gomes -->
<!-- SPDX-License-Identifier: LicenseRef-DLV-Personal-Use -->

Questions that came up while working through the course modules, one section
per notebook in [`tutorials/`](tutorials/).

## 00 — JupyterLab (`00_jupyterlab.ipynb`)

_No questions yet._

## 01 — MNIST (`01_mnist.ipynb`)

### What is the difference between training, validation, and test data?

- **Training data:** the model learns its weights from this data through
  gradient descent.
- **Validation data:** never trained on. During development you use it to tune
  hyperparameters (learning rate, architecture, number of epochs), to spot
  overfitting (training accuracy rising while validation accuracy flattens or
  drops), and to choose a checkpoint or stop early. Because you tune those
  choices against it, its score ends up slightly optimistic.
- **Test data:** kept aside and used once, at the very end, to estimate how
  well the model does on unseen data. It must not influence any decision. If it
  does, it becomes a second validation set and its score is no longer unbiased.

All three should come from the same distribution and must not overlap.
Otherwise the scores look better than they really are.

**In this notebook:** MNIST has 70,000 images. TorchVision splits them into
60,000 for training (`train=True`) and 10,000 in the official *test* split
(`train=False`). The notebook loads that test split as `valid_set` and uses it
for validation after every epoch, so it serves as both validation and test
set. That's fine for a tutorial, but in practice you would split a validation
set off the training data and keep the test split for one final evaluation.

### How should data be split into train / validation / test? Is there a standard?

There is no fixed standard. The split depends mainly on how much data you
have. What matters is that the validation and test sets are big enough to give
stable estimates. Common conventions:

| Dataset size | Typical split (train / val / test) | Why |
| --- | --- | --- |
| Small (≤ ~10k samples) | 60 / 20 / 20 or 70 / 15 / 15 | Larger val/test shares are needed for scores you can trust |
| Medium (~10k–1M) | 80 / 10 / 10 | Common default |
| Large (millions) | 98 / 1 / 1 or smaller | 1% of 1M is still 10,000 samples, enough to evaluate |

Rule of thumb: decide how many validation and test samples you need for
reliable scores (often a few thousand, and enough examples of each class), and
give everything else to training.

Practical points:

- **Very small datasets:** use *k-fold cross-validation* (often k = 5 or 10)
  instead of a single validation split. Each sample is used for validation
  exactly once, and you still keep a separate test set.
- **Imbalanced classes:** use a *stratified* split so each set keeps the same
  class proportions.
- **Time series:** split by time (train on the past, validate and test on
  later periods), never randomly, or future data leaks into training.
- **Grouped data** (e.g. several images of the same person or patient): split
  by group so one group never appears in two sets.
- **Benchmark datasets** come with an official split. Use it so your results
  can be compared with others. MNIST's is 60,000 train / 10,000 test (~86 / 14);
  you would take the validation set out of the 60,000 training images, e.g.
  50,000 / 10,000.

### Why shuffle the training data but not the validation data?

- **Training (`shuffle=True`):** the weights are updated after every batch, so
  order matters. Shuffling makes each batch a mix of all classes and gives
  different batches each epoch. Without it, data stored in some order (e.g.
  sorted by label) would pull the model towards one class at a time.
- **Validation (no shuffle):** no weights change, so the order doesn't affect
  the score. A fixed order keeps results comparable across epochs and makes
  individual samples easy to find again.

### What are layers, and what is calculated between the input and the output?

A layer is one step that turns a list of numbers into a new list of numbers.
The network is these steps chained together: each layer's output is the next
layer's input.

The notebook's model, for one image:

| Step | Layer | Numbers in → out | What it computes |
| --- | --- | --- | --- |
| 1 | `Flatten` | 1×28×28 → 784 | Lays the 28 rows of pixels end to end in one list |
| 2 | `Linear(784, 512)` | 784 → 512 | Each output = weighted sum of all 784 inputs + a bias |
| 3 | `ReLU` | 512 → 512 | Replaces negative numbers with 0 |
| 4 | `Linear(512, 512)` | 512 → 512 | Weighted sums again |
| 5 | `ReLU` | 512 → 512 | Negatives → 0 |
| 6 | `Linear(512, 10)` | 512 → 10 | One score per digit 0–9 |

The prediction is the digit with the highest of the 10 scores.

- **Linear layer:** each output is `w₁·x₁ + w₂·x₂ + … + wₙ·xₙ + b`, where the
  `x` values are the inputs, and each output has its own weights `w` and bias
  `b`. These weights and biases are what training adjusts.
- **ReLU (activation):** `max(0, x)`. Without it, the linear layers would
  collapse into one single weighted sum and the model could only learn simple
  (straight-line) patterns.

### What is a gradient, and why is it −24 in the one-weight example?

The gradient answers one question for each weight: **if I turn this weight up
a tiny bit, does the loss go up or down, and by how much?** It is one number
per weight. `backward()` computes it and stores it in the weight's `.grad`.

Section 1.7.3 uses a model with a single weight `w`. It predicts `w × 3`, the
correct answer is `10`, and the loss is `(prediction − 10)²`. The starting
value `w = 2` is just an easy number for the example. Real models start with
small random weights.

**1. Compute the loss at `w = 2`.** Prediction `2 × 3 = 6`, error
`6 − 10 = −4`, loss `(−4)² = 16`. The gradient is measured on the loss (the
squared error), not on the error itself.

**2. Turn the weight up a tiny bit and compute the loss again.**

| | w | prediction | error | loss |
| --- | --- | --- | --- | --- |
| before | 2.00 | 6.00 | −4.00 | 16.00 |
| after | 2.01 | 6.03 | −3.97 | 15.76 |

**3. Compare.** Turning `w` up by 0.01 made the loss go **down** by 0.24. Per
unit of turning, that is 0.24 ÷ 0.01 = 24. The loss went down, so the
gradient is **−24**.

Measured precisely, the drop is 0.2391, which gives 23.91. Smaller nudges get
closer to 24:

| Nudge | Loss drop | Drop ÷ nudge |
| --- | --- | --- |
| 0.01 | 0.2391 | 23.91 |
| 0.001 | 0.023991 | 23.991 |
| 0.0001 | 0.00239991 | 23.9991 |

The gradient is the value this approaches as the nudge gets infinitely small:
exactly −24.

**Where the 24 comes from.** Follow the 0.01 turn through the model:

1. **Weight → prediction:** the prediction is `w × 3`, so it moves 3 times as
   much as the weight. 0.01 becomes 0.03.
2. **Prediction → loss:** the loss is the error squared. Near an error of 4,
   it changes 2 × 4 = 8 times as much as the error. 0.03 becomes 0.24.

So 0.01 → 0.03 → 0.24 is 3 × 8 = 24 times the turn. That is the formula
`2 × (6 − 10) × 3 = −24`: `2 × (6 − 10)` is the squaring link (8, negative
because a higher prediction lowers the loss), and `× 3` is the weight's effect
on the prediction. `backward()` does exactly this for all 669,706 weights and
biases of the notebook's model: it follows the links back from the loss to
each weight and multiplies them. That is why it is called backpropagation.

**What the sign and size mean:**

- **Negative:** turning the weight up lowers the loss, so `optimizer.step()`
  makes the weight bigger.
- **Positive:** turning the weight up raises the loss, so `step()` makes it
  smaller.
- **Large (ignoring the sign):** the loss is very sensitive to this weight, so
  `step()` moves it further.

You can check it in PyTorch:

```python
w = torch.tensor(2.0, requires_grad=True)
loss = (w * 3 - 10) ** 2
loss.backward()
w.grad   # tensor(-24.)
```

Section 1.7.3 of the notebook shows what `zero_grad()`, `backward()` and
`step()` do with the gradient, with an interactive diagram of this model.

## 02 — ASL (`02_asl.ipynb`)

_No questions yet._

## 03 — ASL with CNNs (`03_asl_cnn.ipynb`)

_No questions yet._

## 04a — ASL data augmentation (`04a_asl_augmentation.ipynb`)

_No questions yet._

## 04b — ASL predictions (`04b_asl_predictions.ipynb`)

_No questions yet._

## 05a — Doggy door (`05a_doggy_door.ipynb`)

_No questions yet._

## 05b — Corgi door (`05b_corgi_door.ipynb`)

_No questions yet._

## 06 — NLP (`06_nlp.ipynb`)

_No questions yet._
