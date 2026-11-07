# SPDX-FileCopyrightText: 2026 Pedro Gomes
# SPDX-License-Identifier: LicenseRef-DLV-Personal-Use

"""Interactive diagrams, quiz and exercise checks for 02_asl.ipynb.

    from interactive import asl as viz
    viz.explore_csv_row(x_train, y_train)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .mnist import (GREY, LIGHT_GREY, PRIMARY, PRIMARY_DARK, PRIMARY_LIGHT, RED, TEXT, _clean,
                    _int_slider, _mosaic, _show, widgets)

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"  # 24 letters: J and Z need motion and are left out
ORANGE = "#E07B39"


def letter(label):
    """The letter for a label from 0 to 23."""
    return LETTERS[int(label)]


def _values(data):
    """NumPy array from a DataFrame, Series, tensor or array."""
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    if hasattr(data, "to_numpy"):
        return data.to_numpy()
    return np.asarray(data)


# ---------------------------------------------------------------------------
# 2.3 Loading the data
# ---------------------------------------------------------------------------

def explore_csv_row(x, y):
    """Show how one CSV row (a label and 784 pixel columns) becomes a 28 x 28 image."""
    x, y = _values(x), _values(y)
    index = _int_slider("CSV row (image)", 0, 0, len(x) - 1, width="560px")
    row = _int_slider("Image row", 10, 0, 27)

    def draw(index, row):
        pixels = x[index].astype(float)
        image = pixels.reshape(28, 28)
        start = row * 28
        fig = plt.figure(figsize=(13, 6.2))
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 3], width_ratios=[1, 2.2], hspace=0.6, wspace=0.18)
        ax_csv = fig.add_subplot(grid[0, :])
        ax_img = fig.add_subplot(grid[1, 0])
        ax_row = fig.add_subplot(grid[1, 1])

        ax_csv.imshow(pixels[None, :], cmap="gray", vmin=0, vmax=255, aspect="auto",
                      extent=(-0.5, 783.5, 0, 1))
        ax_csv.add_patch(Rectangle((start - 0.5, -0.06), 28, 1.12, fill=False, ec=PRIMARY, lw=2.5,
                                   clip_on=False))
        ax_csv.set_yticks([])
        ax_csv.set_xticks([0, 112, 224, 336, 448, 560, 672, 783])
        ax_csv.set_xticklabels([f"pixel{t + 1}" for t in [0, 112, 224, 336, 448, 560, 672, 783]], fontsize=8)
        ax_csv.set_title(f"CSV row {index:,}:  label = {int(y[index])} (letter {letter(y[index])})  +  "
                         "784 pixel columns", fontsize=12, pad=10)

        ax_img.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax_img.add_patch(Rectangle((-0.5, row - 0.5), 28, 1, fill=False, ec=PRIMARY, lw=2.5))
        ax_img.set_xticks([])
        ax_img.set_yticks([0, row, 27])
        ax_img.set_title(f"row.reshape(28, 28)\nletter {letter(y[index])}", fontsize=12)

        ax_row.bar(range(28), image[row], color=PRIMARY)
        ax_row.set_ylim(0, 260)
        ticks = [0, 9, 18, 27]
        ax_row.set_xticks(ticks)
        ax_row.set_xticklabels([f"col {c}\npixel{start + c + 1}" for c in ticks], fontsize=9)
        ax_row.set_ylabel("value (0-255)")
        ax_row.set_title(f"Image row {row} = CSV columns pixel{start + 1} to pixel{start + 28}", fontsize=12)
        _clean(ax_row)
        plt.show()

    _show({"index": index, "row": row}, draw, widgets.VBox([index, row]))


def explore_asl_dataset(x_train, y_train, x_valid, y_valid):
    """Browse the training and validation images with their letters and count the images per letter."""
    sets = {
        f"Training ({len(x_train):,})": (_values(x_train), _values(y_train)),
        f"Validation ({len(x_valid):,})": (_values(x_valid), _values(y_valid)),
    }
    counts = {name: np.bincount(ys.astype(int), minlength=24) for name, (_, ys) in sets.items()}
    split = widgets.ToggleButtons(options=list(sets), description="Data:")
    index = _int_slider("Image number", 0, 0, len(x_train) - 1, width="560px")
    split.observe(lambda change: setattr(index, "max", len(sets[change["new"]][0]) - 1), names="value")

    def draw(split, index):
        xs, ys = sets[split]
        index = min(index, len(xs) - 1)
        label = int(ys[index])
        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 3]})
        ax_img.imshow(xs[index].reshape(28, 28), cmap="gray")
        ax_img.set_title(f"image #{index:,}\nlabel {label} = letter {letter(label)}", fontsize=12)
        ax_img.axis("off")
        colors = [PRIMARY if k == label else LIGHT_GREY for k in range(24)]
        ax_bar.bar(range(24), counts[split], color=colors)
        ax_bar.set_xticks(range(24))
        ax_bar.set_xticklabels(list(LETTERS))
        ax_bar.set_ylabel("number of images")
        ax_bar.set_title(f"{split.split(' (')[0]} set: images per letter", fontsize=12)
        _clean(ax_bar)
        plt.tight_layout()
        plt.show()

    _show({"split": split, "index": index}, draw, widgets.VBox([split, index]))


# ---------------------------------------------------------------------------
# 2.4 Normalising the data
# ---------------------------------------------------------------------------

_SCALES = ["Raw pixels (0 to 255)", "Divided by 255 (0 to 1)", "Standardised (mean 0, std 1)"]


def explore_normalization(x_raw, sample=500, seed=0):
    """Compare raw, divided-by-255 and standardised pixels, and their effect on a neuron's weighted sum."""
    raw = _values(x_raw)[:sample].astype(float)
    mean, std = raw.mean(), raw.std()
    versions = {_SCALES[0]: raw, _SCALES[1]: raw / 255, _SCALES[2]: (raw - mean) / std}
    rng = np.random.default_rng(seed)
    bound = 1 / np.sqrt(784)  # how PyTorch initialises the weights of nn.Linear(784, ...)
    weights = rng.uniform(-bound, bound, size=(784, 512))
    sums = {name: np.abs(values @ weights).mean() for name, values in versions.items()}
    scale = widgets.ToggleButtons(options=_SCALES, description="Pixels:",
                                  style={"button_width": "210px"})

    def draw(scale):
        values = versions[scale]
        fig, (ax_img, ax_hist, ax_sum) = plt.subplots(1, 3, figsize=(14, 4.2),
                                                      gridspec_kw={"width_ratios": [1, 1.6, 1.4]})
        shown = ax_img.imshow(values[0].reshape(28, 28), cmap="gray")
        fig.colorbar(shown, ax=ax_img, fraction=0.046, pad=0.04)
        ax_img.set_title("Same image, new numbers", fontsize=12)
        ax_img.axis("off")
        ax_hist.hist(values.ravel(), bins=40, color=PRIMARY)
        ax_hist.set_title(f"All pixel values: {values.min():.2f} to {values.max():.2f}", fontsize=12)
        ax_hist.set_xlabel("pixel value")
        ax_hist.set_yticks([])
        _clean(ax_hist)
        names = ["raw", "÷ 255", "standardised"]
        colors = [PRIMARY if name == scale else LIGHT_GREY for name in versions]
        bars = ax_sum.bar(names, list(sums.values()), color=colors)
        ax_sum.bar_label(bars, fmt="%.2f", fontsize=9)
        ax_sum.set_yscale("log")
        ax_sum.set_title("Typical size of a neuron's weighted sum\n(first Linear layer, before training)",
                         fontsize=11)
        _clean(ax_sum)
        fig.suptitle("The picture looks the same, but huge inputs make huge weighted sums and gradients, "
                     "which makes training unstable.", fontsize=11)
        plt.tight_layout()
        plt.show()

    _show({"scale": scale}, draw, scale)


# ---------------------------------------------------------------------------
# 2.4.2 Custom datasets and DataLoaders
# ---------------------------------------------------------------------------

def explore_custom_dataset(dataset, loader):
    """What dataset[idx] returns, and how a DataLoader turns random indices into one batch."""
    index = _int_slider("idx", 0, 0, len(dataset) - 1, width="520px")
    clicks = widgets.IntText(value=0, layout=widgets.Layout(display="none"))
    button = widgets.Button(description="Draw a new batch", icon="refresh", button_style="success")
    button.on_click(lambda _: setattr(clicks, "value", clicks.value + 1))
    batch_size = loader.batch_size

    def draw(index, clicks):
        x, y = dataset[index]
        order = list(iter(loader.sampler))[:batch_size]
        items = [dataset[i] for i in order]
        images = np.stack([_values(item[0]).reshape(28, 28) for item in items])
        labels = [int(_values(item[1])) for item in items]

        fig, (ax_item, ax_batch) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 2.6]})
        ax_item.imshow(_values(x).reshape(28, 28), cmap="gray")
        ax_item.set_title(f"dataset[{index}]\nx: shape {tuple(x.shape)}   y: {int(_values(y))} "
                          f"({letter(_values(y))})", fontsize=11)
        ax_item.axis("off")
        _mosaic(images, [letter(lab) for lab in labels], ax_batch,
                f"DataLoader batch #{clicks + 1}: indices {order[:6]} ...\n"
                f"→ x: ({len(order)}, 784)   y: ({len(order)},)")
        fig.suptitle(f"len(dataset) = {len(dataset):,} · batch_size = {batch_size} · "
                     f"{len(loader):,} batches per epoch", fontsize=12)
        plt.tight_layout()
        plt.show()

    _show({"index": index, "clicks": clicks}, draw, widgets.VBox([index, widgets.HBox([button, clicks])]))


# ---------------------------------------------------------------------------
# 2.6.3 Calculating the accuracy
# ---------------------------------------------------------------------------

def explain_batch_accuracy(N=19_200, batch=8):
    """Step through get_batch_accuracy on a small example batch."""
    clicks = widgets.IntText(value=0, layout=widgets.Layout(display="none"))
    button = widgets.Button(description="New example batch", icon="refresh", button_style="success")
    button.on_click(lambda _: setattr(clicks, "value", clicks.value + 1))

    def draw(clicks):
        rng = np.random.default_rng(clicks)
        y = rng.integers(0, 24, size=batch)
        output = rng.normal(0, 1, size=(batch, 24))
        right = rng.random(batch) < 0.75
        output[np.arange(batch)[right], y[right]] += 4
        pred = output.argmax(axis=1)
        correct = int((pred == y).sum())

        fig, (ax_out, ax_tab) = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
        ax_out.imshow(output, cmap="Blues", aspect="auto")
        ax_out.scatter(pred, range(batch), marker="o", s=90, facecolor="none", edgecolor=ORANGE, lw=2)
        ax_out.set_xticks(range(24))
        ax_out.set_xticklabels(list(LETTERS), fontsize=8)
        ax_out.set_yticks(range(batch))
        ax_out.set_yticklabels([f"image {k}" for k in range(batch)])
        ax_out.set_title("output: 24 scores per image (orange circle = output.argmax(dim=1))", fontsize=11)

        ax_tab.axis("off")
        ax_tab.set_xlim(0, 1)
        ax_tab.set_ylim(-1.5, batch + 1)
        for col, head in zip([0.05, 0.35, 0.65], ["pred", "y", "pred.eq(y)"]):
            ax_tab.text(col, batch + 0.3, head, fontsize=11, weight="bold", family="monospace")
        for k in range(batch):
            row = batch - 1 - k
            ok = pred[k] == y[k]
            ax_tab.text(0.05, row, letter(pred[k]), fontsize=11, family="monospace")
            ax_tab.text(0.35, row, letter(y[k]), fontsize=11, family="monospace")
            ax_tab.text(0.65, row, "True" if ok else "False", fontsize=11, family="monospace",
                        color=PRIMARY_DARK if ok else RED)
        ax_tab.text(0.05, -0.9, f".sum().item() = {correct}\n{correct} / N = {correct} / {N:,} = {correct / N:.5f}",
                    fontsize=11, family="monospace", va="top")
        fig.suptitle("Each batch adds correct / N to the epoch's accuracy; after all batches the total is "
                     "the accuracy over the whole dataset.", fontsize=11)
        plt.tight_layout()
        plt.show()

    _show({"clicks": clicks}, draw, widgets.HBox([button, clicks]))


# ---------------------------------------------------------------------------
# 2.6 Training: learning curves and overfitting
# ---------------------------------------------------------------------------

def plot_history(history):
    """Learning curves from the training loop: accuracy or loss per epoch, best epoch and gap."""
    history = np.asarray(history, dtype=float)
    if history.ndim != 2 or len(history) == 0:
        print("Run the training loop first: it fills `history`.")
        return
    epochs = len(history)
    epoch = _int_slider("Up to epoch", epochs, 1, epochs)
    metric = widgets.ToggleButtons(options=["Accuracy", "Loss"], description="Show:")

    def draw(epoch, metric):
        col_train, col_valid = (1, 3) if metric == "Accuracy" else (0, 2)
        xs = np.arange(1, epoch + 1)
        train, valid = history[:epoch, col_train], history[:epoch, col_valid]
        best = int(np.argmax(history[:epoch, 3])) + 1
        fig, ax = plt.subplots(figsize=(12, 4.6))
        ax.plot(xs, train, "o-", color=PRIMARY, label="training")
        ax.plot(xs, valid, "o-", color=ORANGE, label="validation")
        ax.fill_between(xs, train, valid, color=LIGHT_GREY, alpha=0.5, label="gap")
        ax.axvline(best, color=GREY, ls="--", lw=1)
        ax.text(best, ax.get_ylim()[1], f" best validation accuracy: epoch {best}", va="top", fontsize=10,
                color=GREY)
        ax.set_xlim(0.5, epochs + 0.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric.lower())
        gap = history[epoch - 1, 1] - history[epoch - 1, 3]
        ax.set_title(f"Epoch {epoch}: training accuracy {history[epoch - 1, 1]:.1%}, validation accuracy "
                     f"{history[epoch - 1, 3]:.1%}, gap {gap:.1%}", fontsize=12)
        ax.legend(loc="center right")
        _clean(ax)
        plt.show()

    _show({"epoch": epoch, "metric": metric}, draw, widgets.VBox([epoch, metric]))


def explore_overfitting(seed=3):
    """Fit curves of growing flexibility to a few noisy points: underfitting, good fit and overfitting."""
    rng = np.random.default_rng(seed)
    truth = lambda t: np.sin(2 * np.pi * t)  # noqa: E731 - small helper
    x_train = np.sort(rng.uniform(0, 1, 12))
    y_train = truth(x_train) + rng.normal(0, 0.25, 12)
    x_valid = np.sort(rng.uniform(0, 1, 30))
    y_valid = truth(x_valid) + rng.normal(0, 0.25, 30)
    degrees = list(range(0, 10))
    fits = {d: np.polynomial.Polynomial.fit(x_train, y_train, d) for d in degrees}
    err = {d: (np.mean((fits[d](x_train) - y_train) ** 2), np.mean((fits[d](x_valid) - y_valid) ** 2))
           for d in degrees}
    degree = _int_slider("Model flexibility (polynomial degree)", 1, 0, 9, width="460px")

    def draw(degree):
        fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
        grid = np.linspace(0, 1, 300)
        ax.plot(grid, truth(grid), color=LIGHT_GREY, lw=6, label="true pattern")
        ax.plot(grid, np.clip(fits[degree](grid), -3, 3), color=TEXT, lw=2, label="model")
        ax.scatter(x_train, y_train, color=PRIMARY, s=60, zorder=3, label="training data")
        ax.scatter(x_valid, y_valid, color=ORANGE, s=25, zorder=3, label="validation data")
        ax.set_ylim(-2.2, 2.2)
        ax.legend(loc="lower left", fontsize=9)
        _clean(ax)
        train_err, valid_err = err[degree]
        if degree <= 1:
            verdict = "Underfitting: too simple to follow the pattern. Both errors are high."
        elif valid_err <= 1.5 * min(v for _, v in err.values()):
            verdict = "Good fit: follows the pattern, not the noise."
        else:
            verdict = "Overfitting: passes through the training points, but misses new data."
        ax.set_title(verdict, fontsize=12)

        ax_err.plot(degrees, [max(err[d][0], 1e-3) for d in degrees], "o-", color=PRIMARY, label="training error")
        ax_err.plot(degrees, [min(err[d][1], 10) for d in degrees], "o-", color=ORANGE, label="validation error")
        ax_err.axvline(degree, color=GREY, ls="--")
        ax_err.set_yscale("log")
        ax_err.set_ylim(1e-3, 12)
        ax_err.set_xlabel("model flexibility")
        ax_err.set_ylabel("mean squared error")
        ax_err.set_title(f"training {train_err:.3f} · validation {valid_err:.3f}", fontsize=12)
        ax_err.legend(fontsize=9)
        _clean(ax_err)
        plt.tight_layout()
        plt.show()

    _show({"degree": degree}, draw)


# ---------------------------------------------------------------------------
# Check your understanding: certification-style questions
# ---------------------------------------------------------------------------

QUESTIONS = [
    dict(topic="Data analysis",
         question="A CSV file has a 'label' column followed by 784 pixel columns. Which line stores the labels "
                  "in y_train and removes them from the DataFrame, so that only pixels remain?",
         options=["y_train = train_df.pop('label')", "y_train = train_df['label']",
                  "y_train = train_df.values[:, 0]", "y_train = train_df.drop('pixel1')"],
         answer=0,
         explanation="pop() returns the column and removes it from the DataFrame. The other options either "
                     "leave the label among the features or do something else entirely."),
    dict(topic="Data analysis",
         question="x_train[0] is a NumPy array of 784 pixel values. Which call turns it into an image that "
                  "matplotlib can display?",
         options=["x_train[0].reshape(28, 28)", "x_train[0].flatten()", "x_train[0].T", "x_train[0] / 255"],
         answer=0,
         explanation="reshape(28, 28) arranges the 784 values as 28 rows of 28 pixels, which is how they "
                     "were flattened in the first place."),
    dict(topic="Core ML and AI knowledge",
         question="Why does the ASL model have 24 output classes instead of 26?",
         options=["J and Z require hand movement, so they cannot be shown in a single image",
                  "The dataset has no examples of vowels",
                  "The output layer is limited to 24 neurons",
                  "Two letters were merged to balance the dataset"],
         answer=0,
         explanation="J and Z are signed with motion. A still image cannot capture them, so they are left out."),
    dict(topic="Software development",
         question="Which two methods must a custom torch.utils.data.Dataset implement so that a DataLoader "
                  "can use it?",
         options=["__len__ and __getitem__", "__init__ and forward", "train and validate", "__iter__ and __next__"],
         answer=0,
         explanation="The DataLoader asks the dataset for its length and then for individual items by index."),
    dict(topic="Performance optimization",
         question="MyDataset moves all images to the GPU once, in __init__. What is the trade-off?",
         options=["Batches need no transfer during training, but the whole dataset must fit in GPU memory",
                  "Training becomes slower, but uses less GPU memory",
                  "Shuffling is no longer possible",
                  "The model has to run on the CPU"],
         answer=0,
         explanation="Copying once avoids a CPU-to-GPU transfer for every batch, which is fast for small "
                     "datasets like this one. Large datasets do not fit and are moved batch by batch instead."),
    dict(topic="Data analysis",
         question="x_train holds pixel values from 0 to 255. Why does the notebook divide them by 255 before "
                  "training?",
         options=["Small input values keep weighted sums and gradients in a range where training is stable",
                  "It turns the images into grayscale",
                  "It makes the dataset smaller on disk",
                  "PyTorch only accepts values between 0 and 1"],
         answer=0,
         explanation="With raw values, a neuron's weighted sum is hundreds of times larger, and so are its "
                     "gradients. Scaling keeps training well behaved."),
    dict(topic="Software development",
         question="batch = next(iter(train_loader)) with batch_size=32 and the ASL MyDataset. What does batch contain?",
         options=["[x, y] with x of shape (32, 784) and y of shape (32,)",
                  "32 separate (x, y) pairs",
                  "One image of shape (28, 28)",
                  "[x, y] with x of shape (784, 32)"],
         answer=0,
         explanation="The DataLoader stacks the 32 items it drew: images along a new first dimension, and "
                     "labels into one vector."),
    dict(topic="Software development",
         question="get_batch_accuracy returns correct / N, where N is the size of the whole dataset, not of the batch. Why?",
         options=["So that adding up the results of all batches gives the accuracy over the whole epoch",
                  "Because the batch size is unknown inside the function",
                  "To make the accuracy smaller and easier to read",
                  "Because the last batch is always empty"],
         answer=0,
         explanation="Each batch contributes its share of the total. Summed over every batch, the result is "
                     "correct predictions divided by all images."),
    dict(topic="Experimentation",
         question="After 20 epochs, training accuracy is almost 100% but validation accuracy is clearly lower. "
                  "What is this called?",
         options=["Overfitting", "Underfitting", "Vanishing gradients", "Data leakage"],
         answer=0,
         explanation="The model has learned the training images, including their noise, better than the "
                     "general pattern, so it does worse on images it has never seen."),
    dict(topic="Experimentation",
         question="Which of these does NOT help against overfitting?",
         options=["Training for many more epochs",
                  "Collecting more training data",
                  "Data augmentation",
                  "Stopping at the epoch with the best validation accuracy"],
         answer=0,
         explanation="More epochs give the model more time to memorise the training set, which usually makes "
                     "overfitting worse."),
    dict(topic="Core ML and AI knowledge",
         question="The ASL images arrive already flattened, with shape (32, 784). What does nn.Flatten() do to them?",
         options=["Nothing: they stay (32, 784)", "It turns them into (25088,)",
                  "It turns them into (32, 28, 28)", "It raises an error"],
         answer=0,
         explanation="nn.Flatten keeps the batch dimension and flattens the rest. Already-flat data passes "
                     "through unchanged, so the same model works for both datasets."),
    dict(topic="Experimentation",
         question="Validation loss reaches its lowest value at epoch 6 and then rises, while training loss keeps "
                  "falling. Which model should you keep?",
         options=["The model from epoch 6", "The model from the last epoch",
                  "The model with the lowest training loss", "Any of them: they are all equally good"],
         answer=0,
         explanation="This is early stopping: keep the model that does best on unseen data. Later epochs only "
                     "fit the training set more closely."),
]


def quiz():
    """Certification-style multiple-choice questions about this notebook."""
    from .quiz import run_quiz
    run_quiz(QUESTIONS, seed=35)


# ---------------------------------------------------------------------------
# Practice exercises: automatic checks
# ---------------------------------------------------------------------------

def check_letter_function(letter_of):
    from .quiz import failed, passed
    try:
        got = [letter_of(k) for k in range(24)]
    except Exception as error:  # noqa: BLE001 - show any error to the student
        failed(f"Your function raised an error: {error}")
        return
    if got == list(LETTERS):
        passed("All 24 labels map to the right letter, skipping J and Z.")
    elif None in got:
        failed("The function returns None. Did you forget return?")
    else:
        wrong = next(k for k in range(24) if got[k] != LETTERS[k])
        failed(f"Label {wrong} should be '{LETTERS[wrong]}', not {got[wrong]!r}. Remember that J is missing.")


def check_counts(counts, y_train):
    from .quiz import failed, passed
    if counts is None:
        failed("Replace None with your list of 24 counts.")
        return
    expected = np.bincount(_values(y_train).astype(int), minlength=24).tolist()
    values = [int(c) for c in _values(counts)]
    if values == expected:
        passed(f"Every letter has {expected[0]} training images: the classes are perfectly balanced, "
               "so accuracy is a fair measure here.")
    elif len(values) != 24:
        failed(f"Expected 24 counts (one per letter), got {len(values)}.")
    else:
        failed("Some counts are off. Make sure they are sorted by label, from 0 (A) to 23 (Y).")


def check_image_dataset(dataset_class, x, y):
    import torch
    from .quiz import failed, passed
    if dataset_class is None:
        failed("Replace None with your class.")
        return
    try:
        data = dataset_class(x, y)
        image, label = data[3]
    except Exception as error:  # noqa: BLE001 - show any error to the student
        failed(f"Creating the dataset or reading an item failed: {error}")
        return
    if len(data) != len(x):
        failed(f"len() should be {len(x):,}, not {len(data):,}.")
    elif not torch.is_tensor(image) or tuple(image.shape) != (1, 28, 28):
        shape = tuple(image.shape) if hasattr(image, "shape") else type(image).__name__
        failed(f"Each image should be a tensor of shape (1, 28, 28), not {shape}.")
    elif image.dtype != torch.float32:
        failed(f"Images should be float32 tensors, not {image.dtype}.")
    elif int(label) != int(_values(y)[3]):
        failed("The label does not match. __getitem__ should return (image, label) for the same index.")
    else:
        passed("Each item is a 1 × 28 × 28 image and its label: the format the CNN in lesson 03 expects.")


def check_history_analysis(best_epoch, final_gap, history):
    from .quiz import failed, passed
    if best_epoch is None or final_gap is None:
        failed("Replace both None values.")
        return
    history = np.asarray(history, dtype=float)
    expected_best = int(np.argmax(history[:, 3])) + 1
    expected_gap = history[-1, 1] - history[-1, 3]
    if int(best_epoch) == expected_best - 1:
        failed("Off by one: epochs are counted from 1 in the training printout and the plot.")
    elif int(best_epoch) != expected_best:
        failed(f"The best epoch is the one with the highest validation accuracy (column 3 of history).")
    elif abs(float(final_gap) - expected_gap) > 1e-6:
        failed("final_gap should be the last training accuracy minus the last validation accuracy.")
    elif expected_gap < 0.01:
        passed(f"Best epoch: {expected_best}. Final gap: {expected_gap:.1%}. In this run, validation accuracy "
               "is about as high as training accuracy, so there is almost no overfitting (see the note in "
               "section 2.6.4).")
    else:
        passed(f"Best epoch: {expected_best}. Final gap: {expected_gap:.1%}. Training past epoch "
               f"{expected_best} did not help on new data.")
