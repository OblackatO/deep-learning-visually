# Lesson 01 on one page: image classification with MNIST

**The idea.** Traditional programming: a person writes the rules. Deep learning: the model *learns* the
rules from many **labelled examples** (`X` = images, `Y` = labels). MNIST: 70,000 grayscale 28 × 28
digits, **60,000 for training** (the model learns from them) and **10,000 for validation** (never trained
on; they show how well the model **generalises**). Accuracy measured on training data proves nothing.

## 1 · Data → tensors → batches

| Step | Code | Result |
|---|---|---|
| Load | `torchvision.datasets.MNIST("./data/", train=True, download=True)` | `(PIL image, int label)` pairs |
| To tensor | `transforms.Compose([ToImage(), ToDtype(torch.float32, scale=True)])` | shape **`(1, 28, 28)`** = **C × H × W**, values **0.0–1.0** (`scale=True` ÷ 255) |
| Batch | `DataLoader(train_set, batch_size=32, shuffle=True)` | `x`: **`(32, 1, 28, 28)`**, `y`: `(32,)` · 60,000 / 32 = 1,875 batches = 1 **epoch** |
| Device | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` | `.to(device)` works everywhere; `.cuda()` fails without a GPU |

**Shuffle the training data only:** weights change after every batch, so order matters there; validation
changes no weights, so its order does not affect the score.

## 2 · The model

```python
model = nn.Sequential(
    nn.Flatten(),                         # (32, 1, 28, 28) → (32, 784): keeps the batch dim
    nn.Linear(784, 512), nn.ReLU(),       # input layer:  784 × 512 + 512 = 401,920 params
    nn.Linear(512, 512), nn.ReLU(),       # hidden layer: 512 × 512 + 512 = 262,656
    nn.Linear(512, 10),                   # output layer: 512 × 10 + 10   =   5,130 → total 669,706
).to(device)
model = torch.compile(model)              # optional speed-up (PyTorch 2.x)
```

- **Neuron** = weighted sum of all inputs + bias, then an activation. **Parameters of `Linear(in, out)` = in × out + out.**
- **ReLU** (`max(0, x)`) between layers: without a non-linearity, stacked `Linear` layers collapse into one linear function.
- **Output layer: 10 raw scores (logits), no activation**, because `nn.CrossEntropyLoss` applies softmax itself.

## 3 · Training: loss, gradient, optimizer

```python
loss_function = nn.CrossEntropyLoss()           # logits + class indices → one number: how wrong
optimizer = Adam(model.parameters(), lr=0.001)  # lr = step size; 0.001 is Adam's default

model.train()
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    output = model(x)                    # 1. forward pass
    optimizer.zero_grad()                # 2. wipe old gradients (backward() ADDS to .grad)
    batch_loss = loss_function(output, y)  # 3. loss
    batch_loss.backward()                # 4. backpropagation: fills every weight's .grad
    optimizer.step()                     # 5. the ONLY line that changes the weights
```

- **Gradient** = slope of the loss for one weight: *if I nudge this weight up, does the loss rise or fall?*
  Negative → increase the weight, positive → decrease it. **SGD:** `w ← w − lr × gradient`.
  Example: `w = 2`, `w.grad = −24`, `lr = 0.01` → `w = 2.24`. Adam = same idea + a per-weight step size.
- **Learning rate too high** → steps overshoot the minimum, loss jumps or grows. **Too low** → slow.
- **Forgetting `zero_grad()`** → gradients of all previous batches pile up and training goes wrong.

## 4 · Validation and accuracy

```python
correct = 0
model.eval()                             # inference behaviour (dropout, batch norm)
with torch.no_grad():                    # no gradients needed: saves memory and time
    for x, y in valid_loader:
        pred = model(x.to(device)).argmax(dim=1)   # predicted class = index of the highest score
        correct += (pred.cpu() == y).sum().item()
accuracy = correct / len(valid_loader.dataset)
```

**Overfitting:** training accuracy keeps rising while validation accuracy stalls or drops (a growing
gap). The model memorises instead of generalising. Fixes come later: more data, augmentation, stopping earlier.

## Exam traps, at a glance

| If the question says ... | Answer |
|---|---|
| `nn.Flatten()` on `(32, 1, 28, 28)` | `(32, 784)` |
| Parameters of `nn.Linear(784, 512)` | **401,920** (don't forget the 512 biases) |
| Shape/range after `ToImage()` + `ToDtype(float32, scale=True)` | `(1, 28, 28)`, 0.0–1.0 |
| Softmax before `nn.CrossEntropyLoss`? | No: it expects raw logits |
| Which line updates the weights? | `optimizer.step()` (`backward()` only computes gradients) |
| Why `model.eval()` + `torch.no_grad()` in validation? | Inference mode; no gradient tracking |
| 99.9 % accuracy on the training images | Says nothing about new data: use the validation set |
| Loss oscillates or explodes | Learning rate too high |
| Train ↑, validation ↓ | Overfitting |

*Part of [Deep Learning, Visually](../../README.md), adapted from NVIDIA's Fundamentals of Deep Learning.
Test yourself with the quiz and the exercises at the end of
[`01_mnist.ipynb`](../tutorials/01_mnist.ipynb).*
