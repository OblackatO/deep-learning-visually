# SPDX-FileCopyrightText: 2026 Pedro Gomes
# SPDX-License-Identifier: LicenseRef-DLV-Personal-Use

"""Interactive diagrams for 01_mnist.ipynb.

Each function draws a diagram with Matplotlib. Most of them also show
ipywidgets controls (sliders, buttons, ...) so the diagram updates while you
move them. Call them from the notebook, for example:

    from interactive import mnist as viz
    viz.explore_pixels(x_0_tensor)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError as error:  # pragma: no cover - depends on the environment
    raise ImportError(
        "The interactive diagrams need ipywidgets. Install it with: pip install ipywidgets"
    ) from error


PRIMARY = "#0F8B8D"
PRIMARY_DARK = "#0B5F61"
PRIMARY_LIGHT = "#DDF3F3"
TEXT = "#1A1A1A"
GREY = "#8C8C8C"
LIGHT_GREY = "#D0D0D0"
RED = "#D64545"
DIGITS = list(range(10))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _int_slider(description, value, low, high, step=1, width="420px"):
    return widgets.IntSlider(
        value=value, min=low, max=high, step=step, description=description,
        continuous_update=True, style={"description_width": "initial"},
        layout=widgets.Layout(width=width),
    )


def _float_slider(description, value, low, high, step, width="300px", fmt=".2f"):
    return widgets.FloatSlider(
        value=value, min=low, max=high, step=step, description=description,
        readout_format=fmt, continuous_update=True,
        style={"description_width": "initial"}, layout=widgets.Layout(width=width),
    )


def _show(controls, draw, ui=None):
    """Display the controls and a figure that is redrawn when they change."""
    out = widgets.interactive_output(draw, controls)
    display(widgets.VBox([ui if ui is not None else widgets.HBox(list(controls.values())), out]))


def _show_button(button, draw):
    """Display a button that redraws the figure on every click."""
    out = widgets.Output()

    def redraw(_=None):
        with out:
            out.clear_output(wait=True)
            draw()

    button.on_click(redraw)
    display(widgets.VBox([button, out]))
    redraw()


def _grid(image):
    """Return a 28 x 28 float array (0.0-1.0) from a tensor, array or PIL image."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image).squeeze()
    if array.dtype == np.uint8:
        array = array / 255.0
    return array.astype(float)


def _canvas(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, text, fc=PRIMARY_LIGHT, ec=PRIMARY, color=TEXT, size=11, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=1.6,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=color, weight=weight)


def _arrow(ax, start, end, color=GREY, lw=2.0, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=18, color=color, lw=lw,
        shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}",
    ))


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# 1.2 Traditional programming vs deep learning
# ---------------------------------------------------------------------------

def show_rules_vs_learning():
    """Diagram: who writes the rules in traditional programming vs deep learning."""
    fig, ax = _canvas((11, 4.3), (0, 11), (0.25, 4.75))
    rows = [
        (3.0, "Traditional programming", "A person has to work out and write every rule.",
         ("Rules\n(written by a person)", "Data"), "Program", "Answers", False),
        (0.4, "Deep learning", "The model works out the rules from labelled examples.",
         ("Data\n(images)", "Answers\n(labels)"), "Training", "Rules\n(learned weights)", True),
    ]
    for y, title, note, inputs, step, result, learned in rows:
        ax.text(0.1, y + 1.55, title, fontsize=14, weight="bold", color=TEXT)
        ax.text(4.1, y + 1.55, note, fontsize=11, color=GREY, style="italic")
        _box(ax, 0.1, y + 0.7, 2.6, 0.65, inputs[0])
        _box(ax, 0.1, y - 0.05, 2.6, 0.65, inputs[1])
        _box(ax, 3.8, y + 0.2, 2.2, 0.9, step, fc=TEXT, ec=TEXT, color="white", size=12, weight="bold")
        _arrow(ax, (2.75, y + 1.02), (3.75, y + 0.75))
        _arrow(ax, (2.75, y + 0.28), (3.75, y + 0.55))
        _arrow(ax, (6.05, y + 0.65), (7.05, y + 0.65))
        _box(ax, 7.1, y + 0.2, 3.0, 0.9, result,
             fc=PRIMARY if learned else PRIMARY_LIGHT, color="white" if learned else TEXT,
             size=12, weight="bold" if learned else "normal")
    ax.plot([0.1, 10.9], [2.65, 2.65], color=LIGHT_GREY, lw=1)
    plt.show()


# ---------------------------------------------------------------------------
# 1.3 The MNIST dataset
# ---------------------------------------------------------------------------

def explore_dataset(train_set, valid_set):
    """Browse the training and validation images and see how many there are of each digit."""
    sets = {f"Training ({len(train_set):,})": train_set, f"Validation ({len(valid_set):,})": valid_set}
    counts = {name: np.bincount(np.asarray(ds.targets), minlength=10) for name, ds in sets.items()}

    split = widgets.ToggleButtons(options=list(sets), description="Data:")
    index = _int_slider("Image number", 0, 0, len(train_set) - 1, width="560px")

    def on_split(change):
        index.max = len(sets[change["new"]]) - 1

    split.observe(on_split, names="value")

    def draw(split, index):
        ds = sets[split]
        index = min(index, len(ds) - 1)
        image, label = _grid(ds.data[index]), int(ds.targets[index])

        fig, (ax_img, ax_bar) = plt.subplots(
            1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 2]})
        ax_img.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax_img.set_title(f"x = image #{index:,}\ny = label {label}", fontsize=12)
        ax_img.axis("off")

        colors = [PRIMARY if d == label else LIGHT_GREY for d in DIGITS]
        bars = ax_bar.bar(DIGITS, counts[split], color=colors)
        ax_bar.bar_label(bars, fontsize=8, color=GREY)
        ax_bar.set_xticks(DIGITS)
        ax_bar.set_xlabel("digit (label y)")
        ax_bar.set_ylabel("number of images")
        ax_bar.set_title(f"{split.split(' (')[0]} set: images per digit", fontsize=12)
        _clean(ax_bar)
        plt.tight_layout()
        plt.show()

    _show({"split": split, "index": index}, draw, widgets.VBox([split, index]))


# ---------------------------------------------------------------------------
# 1.4 Tensors
# ---------------------------------------------------------------------------

def explore_pixels(image, name="x_0_tensor"):
    """Pick a pixel with the sliders and zoom in on the numbers around it."""
    pixels = _grid(image)
    size = pixels.shape[0]
    row = _int_slider("Row (height)", 6, 0, size - 1)
    col = _int_slider("Column (width)", 13, 0, size - 1)

    def draw(row, col):
        r0 = int(np.clip(row - 3, 0, size - 7))
        c0 = int(np.clip(col - 3, 0, size - 7))
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5.4))

        shown = ax_full.imshow(pixels, cmap="gray", vmin=0, vmax=1)
        ax_full.set_xticks(np.arange(-0.5, size, 1), minor=True)
        ax_full.set_yticks(np.arange(-0.5, size, 1), minor=True)
        ax_full.grid(which="minor", color="#444444", lw=0.3)
        ax_full.tick_params(which="minor", length=0)
        ax_full.set_xticks(range(0, size, 4))
        ax_full.set_yticks(range(0, size, 4))
        ax_full.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), 7, 7, fill=False, ec=PRIMARY, lw=2.5))
        ax_full.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, ec=RED, lw=2))
        ax_full.set_xlabel("column (width)")
        ax_full.set_ylabel("row (height)")
        ax_full.set_title(f"The whole image: {size} × {size} pixels", fontsize=12)
        bar = fig.colorbar(shown, ax=ax_full, fraction=0.046, pad=0.04, ticks=[0, 0.5, 1])
        bar.ax.set_yticklabels(["0.0 black", "0.5 grey", "1.0 white"])

        window = pixels[r0:r0 + 7, c0:c0 + 7]
        ax_zoom.imshow(window, cmap="gray", vmin=0, vmax=1,
                       extent=(c0 - 0.5, c0 + 6.5, r0 + 6.5, r0 - 0.5))
        for i in range(7):
            for j in range(7):
                value = window[i, j]
                ax_zoom.text(c0 + j, r0 + i, f"{value:.2f}", ha="center", va="center",
                             fontsize=9, color="black" if value > 0.5 else "white")
        ax_zoom.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, ec=RED, lw=3))
        ax_zoom.set_xticks(range(c0, c0 + 7))
        ax_zoom.set_yticks(range(r0, r0 + 7))
        ax_zoom.set_xlabel("column")
        ax_zoom.set_ylabel("row")
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor(PRIMARY)
            spine.set_linewidth(2.5)
        value = pixels[row, col]
        ax_zoom.set_title(f"Zoom (outlined square)\n{name}[0, {row}, {col}] = {value:.4f}"
                          f"  (= {round(value * 255)} / 255)", fontsize=12)
        plt.tight_layout()
        plt.show()

    _show({"row": row, "col": col}, draw, widgets.VBox([row, col]))


# ---------------------------------------------------------------------------
# 1.5 DataLoaders
# ---------------------------------------------------------------------------

def _mosaic(images, labels, ax, title, columns=8, pad=2):
    count = len(images)
    rows = int(np.ceil(count / columns))
    side = images.shape[-1]
    canvas = np.full((rows * (side + pad) + pad, columns * (side + pad) + pad), 0.35)
    for k in range(count):
        r, c = divmod(k, columns)
        top, left = pad + r * (side + pad), pad + c * (side + pad)
        canvas[top:top + side, left:left + side] = images[k]
        ax.text(left + 1.5, top + 1.5, str(int(labels[k])), color=PRIMARY,
                fontsize=10, weight="bold", ha="left", va="top")
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def explore_batches(train_loader, valid_loader, max_images=32):
    """Draw the first batch from each DataLoader. Click again to see what changes."""
    clicks = {"count": 0}
    button = widgets.Button(description="Draw the first batch again", icon="refresh",
                            button_style="success", layout=widgets.Layout(width="260px"))

    def draw():
        clicks["count"] += 1
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        for ax, loader, title in [
            (axes[0], train_loader, "train_loader (shuffle=True)\na new mix every time"),
            (axes[1], valid_loader, "valid_loader (no shuffle)\nalways the same batch"),
        ]:
            x, y = next(iter(loader))
            images = _grid(x[:max_images]).reshape(-1, x.shape[-2], x.shape[-1])
            _mosaic(images, y[:max_images], ax, title)
        fig.suptitle(f"Draw #{clicks['count']}: the first {min(max_images, len(x))} images and "
                     "labels of each DataLoader", fontsize=12)
        plt.tight_layout()
        plt.show()

    _show_button(button, draw)


# ---------------------------------------------------------------------------
# 1.6 Creating the model
# ---------------------------------------------------------------------------

_PIPELINE = [
    ("Image", "1 × 28 × 28", "The input: 28 × 28 grayscale pixels with values from 0.0 (black) to 1.0 (white)."),
    ("Flatten", "784", "nn.Flatten() lays the 28 rows end to end: one list of 28 × 28 = 784 numbers."),
    ("Linear + ReLU", "512", "512 neurons. Each one computes a weighted sum of all 784 numbers plus a bias, "
                             "then ReLU turns negative results into 0."),
    ("Linear + ReLU ", "512", "512 neurons again. Each one now looks at the 512 numbers from the previous layer."),
    ("Linear", "10", "10 output neurons, one per digit 0-9. Each produces a score: higher means "
                     "'more likely this digit'."),
    ("argmax", "1 digit", "The prediction is the digit whose score is the highest."),
]


def show_pipeline(image=None):
    """The path of one image through the model. Click a step to see what happens there."""
    picture = None if image is None else _grid(image)
    step = widgets.ToggleButtons(options=[name for name, _, _ in _PIPELINE], description="Step:",
                                 style={"button_width": "110px"})

    def draw(step):
        current = [name for name, _, _ in _PIPELINE].index(step)
        fig, ax = _canvas((13, 3.3), (0, 13), (0, 3.3))
        width, gap = 1.75, 0.45
        for k, (name, shape, _) in enumerate(_PIPELINE):
            x = 0.1 + k * (width + gap)
            active = k == current
            fc = PRIMARY if active else PRIMARY_LIGHT
            color = "white" if active else TEXT
            _box(ax, x, 1.4, width, 1.6, "", fc=fc)
            ax.text(x + width / 2, 2.72, name.strip(), ha="center", va="center",
                    fontsize=11, weight="bold", color=color)
            if k == 0 and picture is not None:
                ax.imshow(picture, cmap="gray", extent=(x + 0.45, x + width - 0.45, 1.5, 2.45), zorder=3)
            elif k == len(_PIPELINE) - 1:
                ax.text(x + width / 2, 1.95, "5", ha="center", va="center",
                        fontsize=26, weight="bold", color=color)
            else:
                ax.text(x + width / 2, 1.95, f"{shape}\nnumbers", ha="center", va="center",
                        fontsize=11, color=color)
            ax.text(x + width / 2, 1.1, f"shape: {shape}", ha="center", fontsize=9, color=GREY)
            if k < len(_PIPELINE) - 1:
                _arrow(ax, (x + width + 0.03, 2.2), (x + width + gap - 0.03, 2.2))
        ax.text(0.1, 0.45, _PIPELINE[current][2], fontsize=12, color=TEXT, wrap=True)
        ax.text(0.1, 0.0, "During training, a whole batch flows through at once: shapes become "
                "(32, 1, 28, 28) → (32, 784) → (32, 512) → (32, 512) → (32, 10).",
                fontsize=10, color=GREY)
        ax.set_ylim(-0.15, 3.3)
        ax.set_aspect("equal")
        plt.show()

    _show({"step": step}, draw, step)


def explore_flatten(image):
    """See where each row of the image ends up after nn.Flatten()."""
    pixels = _grid(image)
    size = pixels.shape[0]
    flat = pixels.reshape(-1)
    row = _int_slider("Row", 8, 0, size - 1)

    def draw(row):
        start, end = row * size, row * size + size - 1
        fig = plt.figure(figsize=(12, 6.2))
        grid = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 2.3],
                                hspace=0.55, wspace=0.18)
        ax_img = fig.add_subplot(grid[0, 0])
        ax_row = fig.add_subplot(grid[0, 1])
        ax_flat = fig.add_subplot(grid[1, :])

        ax_img.imshow(pixels, cmap="gray", vmin=0, vmax=1)
        ax_img.add_patch(Rectangle((-0.5, row - 0.5), size, 1, fill=False, ec=PRIMARY, lw=2.5))
        ax_img.set_title(f"Before: {size} × {size} grid\n(row {row} outlined)", fontsize=12)
        ax_img.set_xticks([])
        ax_img.set_yticks([0, row, size - 1])

        ax_row.bar(range(size), pixels[row], color=PRIMARY)
        ax_row.set_ylim(0, 1.05)
        ticks = [0, 7, 14, 21, size - 1]
        ax_row.set_xticks(ticks)
        ax_row.set_xticklabels([f"col {c}\n→ {start + c}" for c in ticks])
        ax_row.set_ylabel("brightness")
        ax_row.set_title(f"Row {row}: {size} values → positions {start} to {end} after flattening",
                         fontsize=12)
        _clean(ax_row)

        ax_flat.imshow(flat[None, :], cmap="gray", vmin=0, vmax=1, aspect="auto",
                       extent=(-0.5, flat.size - 0.5, 0, 1))
        ax_flat.add_patch(Rectangle((start - 0.5, -0.06), size, 1.12, fill=False,
                                    ec=PRIMARY, lw=2.5, clip_on=False))
        ax_flat.set_yticks([])
        ax_flat.set_xticks(range(0, flat.size + 1, 4 * size))
        ax_flat.set_xlabel("position in the flattened list")
        ax_flat.set_title(f"After nn.Flatten(): one line of {flat.size} numbers", fontsize=12, pad=12)
        plt.show()

    _show({"row": row}, draw)


def explore_neuron():
    """One neuron: weighted sum of its inputs, plus a bias, then ReLU."""
    xs = [_float_slider(f"x{i} (input)", v, 0.0, 1.0, 0.05) for i, v in enumerate([0.9, 0.2, 0.6], 1)]
    ws = [_float_slider(f"w{i} (weight)", v, -2.0, 2.0, 0.1, fmt=".1f")
          for i, v in enumerate([0.8, -1.0, 0.5], 1)]
    b = _float_slider("b (bias)", -0.2, -2.0, 2.0, 0.1, fmt=".1f")

    def draw(x1, x2, x3, w1, w2, w3, b):
        x, w = [x1, x2, x3], [w1, w2, w3]
        z = sum(xi * wi for xi, wi in zip(x, w)) + b
        a = max(0.0, z)

        fig, (ax, ax_relu) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.7, 1]})
        ax.set_xlim(0, 10.4)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")
        center = (5.2, 3.0)
        for i, (xi, wi, y) in enumerate(zip(x, w, [5.0, 3.0, 1.0]), 1):
            ax.add_patch(Circle((0.9, y), 0.55, fc=str(xi), ec=TEXT, lw=1.2, zorder=3))
            ax.text(0.9, y, f"x{i}\n{xi:.2f}", ha="center", va="center", fontsize=10,
                    color="black" if xi > 0.5 else "white", zorder=4)
            color = PRIMARY if wi >= 0 else RED
            ax.plot([1.45, center[0] - 0.8], [y, center[1]], color=color, lw=0.6 + 3 * abs(wi), zorder=1)
            mid = (1.45 + center[0] - 0.8) / 2, (y + center[1]) / 2
            ax.text(mid[0], mid[1] + 0.3, f"w{i} = {wi:+.1f}", ha="center", fontsize=10, color=color,
                    bbox=dict(fc="white", ec="none", pad=1))
        ax.add_patch(Circle(center, 0.8, fc=PRIMARY_LIGHT, ec=PRIMARY, lw=2, zorder=3))
        ax.text(*center, f"Σ + b\nz = {z:.2f}", ha="center", va="center", fontsize=10, zorder=4)
        ax.text(center[0], 5.2, f"bias b = {b:+.1f}", ha="center", fontsize=10)
        _arrow(ax, (center[0], 4.95), (center[0], 3.85))
        _box(ax, 6.7, 2.45, 1.4, 1.1, "ReLU", fc=TEXT, ec=TEXT, color="white", weight="bold")
        _arrow(ax, (center[0] + 0.82, 3.0), (6.65, 3.0))
        _arrow(ax, (8.15, 3.0), (8.8, 3.0))
        ax.add_patch(Circle((9.5, 3.0), 0.65, fc=PRIMARY if a > 0 else LIGHT_GREY, ec=TEXT, lw=1.2))
        ax.text(9.5, 3.0, f"out\n{a:.2f}", ha="center", va="center", fontsize=10,
                color="white" if a > 0 else TEXT)
        ax.set_title(f"z = {w1:+.1f}·{x1:.2f} {w2:+.1f}·{x2:.2f} {w3:+.1f}·{x3:.2f} {b:+.1f} = {z:.2f}"
                     f"     →     ReLU(z) = max(0, z) = {a:.2f}", fontsize=12)

        low, high = min(-3.0, z - 0.5), max(3.0, z + 0.5)
        zs = np.linspace(low, high, 200)
        ax_relu.plot(zs, np.maximum(0, zs), color=PRIMARY, lw=3)
        ax_relu.axhline(0, color=LIGHT_GREY, lw=1)
        ax_relu.axvline(0, color=LIGHT_GREY, lw=1)
        ax_relu.plot([z, z], [0, a], ls="--", color=GREY)
        ax_relu.plot(z, a, "o", ms=11, color=RED if z < 0 else PRIMARY_DARK)
        ax_relu.set_xlabel("z (weighted sum + bias)")
        ax_relu.set_ylabel("output = ReLU(z)")
        ax_relu.set_title("ReLU: negative → 0, positive → unchanged", fontsize=12)
        _clean(ax_relu)
        plt.tight_layout()
        plt.show()

    controls = {"x1": xs[0], "x2": xs[1], "x3": xs[2], "w1": ws[0], "w2": ws[1], "w3": ws[2], "b": b}
    ui = widgets.HBox([widgets.VBox(xs), widgets.VBox(ws + [b])])
    _show(controls, draw, ui)


def _linear_sizes(input_size, hidden, layers, n_classes):
    sizes = [input_size] + [hidden] * layers + [n_classes]
    return list(zip(sizes[:-1], sizes[1:]))


def show_network(input_size=784, n_classes=10):
    """The model's layers as a diagram, with the number of weights and biases in each."""
    hidden = _int_slider("Neurons per layer", 512, 16, 1024, step=16)
    layers = _int_slider("Layers of neurons before the output", 2, 1, 4)

    def draw(hidden, layers):
        sizes = [input_size] + [hidden] * layers + [n_classes]
        fig, (ax, ax_text) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.5, 1]})
        ax.axis("off")
        ax.set_xlim(-0.6, len(sizes) - 0.4)
        ax.set_ylim(-1.9, 10.6)
        positions = []
        for k, size in enumerate(sizes):
            shown = min(size, 10 if k == len(sizes) - 1 else 8)
            ys = np.linspace(9.6, 0.4, shown) if shown > 1 else np.array([5.0])
            positions.append(ys)
            if size > shown:
                ax.text(k, -0.35, "⋮", ha="center", va="center", fontsize=16, color=GREY)
        for k in range(len(sizes) - 1):
            for y0 in positions[k]:
                for y1 in positions[k + 1]:
                    ax.plot([k, k + 1], [y0, y1], color=GREY, lw=0.4, alpha=0.35, zorder=1)
        for k, ys in enumerate(positions):
            is_input, is_output = k == 0, k == len(sizes) - 1
            color = LIGHT_GREY if is_input else (PRIMARY_DARK if is_output else PRIMARY)
            ax.scatter([k] * len(ys), ys, s=170, color=color, edgecolor=TEXT, zorder=3)
            if is_output:
                for digit, y in enumerate(ys):
                    ax.text(k + 0.22, y, str(digit), va="center", fontsize=9)
            label = ("input\n(pixels)" if is_input else "output\n(scores)" if is_output
                     else f"layer {k}\nLinear + ReLU")
            ax.text(k, -1.25, f"{sizes[k]:,}\n{label}", ha="center", va="center", fontsize=10)
        ax.set_title("Every neuron is connected to every neuron in the next layer", fontsize=12)

        ax_text.axis("off")
        lines, total = [], 0
        for n_in, n_out in _linear_sizes(input_size, hidden, layers, n_classes):
            count = n_in * n_out + n_out
            total += count
            lines.append(f"Linear({n_in}, {n_out})\n  {n_in}×{n_out} weights + {n_out} biases = {count:,}")
        code = ["model = nn.Sequential(", "    nn.Flatten(),"]
        for n_in, n_out in _linear_sizes(input_size, hidden, layers, n_classes)[:-1]:
            code.append(f"    nn.Linear({n_in}, {n_out}), nn.ReLU(),")
        code += [f"    nn.Linear({hidden}, {n_classes}),", ")"]
        ax_text.text(0, 1, "Numbers the model learns", fontsize=12, weight="bold", va="top")
        ax_text.text(0, 0.92, "\n".join(lines), fontsize=10, family="monospace", va="top")
        ax_text.text(0, 0.90 - 0.078 * len(lines), f"Total: {total:,} parameters",
                     fontsize=12, weight="bold", color=PRIMARY_DARK, va="top")
        ax_text.text(0, 0.30, "The same model in PyTorch", fontsize=12, weight="bold", va="top")
        ax_text.text(0, 0.23, "\n".join(code), fontsize=9.5, family="monospace", va="top")
        plt.tight_layout()
        plt.show()

    _show({"hidden": hidden, "layers": layers}, draw, widgets.VBox([hidden, layers]))


# ---------------------------------------------------------------------------
# 1.7 Training the model
# ---------------------------------------------------------------------------

def _softmax(scores):
    exps = np.exp(scores - np.max(scores))
    return exps / exps.sum()


def explore_loss():
    """Change the model's 10 output scores and watch the probabilities and the loss."""
    start = {3: 1.0, 5: 2.5}
    scores = [
        widgets.FloatSlider(value=start.get(d, 0.0), min=-5, max=5, step=0.1, description=str(d),
                            orientation="vertical", readout_format=".1f",
                            layout=widgets.Layout(height="170px", width="52px"))
        for d in DIGITS
    ]
    target = widgets.Dropdown(options=DIGITS, value=5, description="Correct digit y:",
                              style={"description_width": "initial"}, layout=widgets.Layout(width="200px"))

    def draw(target, **values):
        logits = np.array([values[f"s{d}"] for d in DIGITS])
        probs = _softmax(logits)
        loss = -np.log(probs[target])
        guess = int(np.argmax(probs))

        fig, (ax_s, ax_p, ax_l) = plt.subplots(1, 3, figsize=(14, 4.2),
                                              gridspec_kw={"width_ratios": [1, 1, 0.9]})
        colors = [PRIMARY if d == target else LIGHT_GREY for d in DIGITS]
        ax_s.bar(DIGITS, logits, color=colors)
        ax_s.axhline(0, color=GREY, lw=1)
        ax_s.set_ylim(-5.5, 5.5)
        ax_s.set_title("1. Model output: 10 scores", fontsize=12)
        ax_p.bar(DIGITS, probs, color=colors)
        for d, p in zip(DIGITS, probs):
            if p >= 0.01:
                ax_p.text(d, p + 0.02, f"{p:.0%}", ha="center", fontsize=8)
        ax_p.set_ylim(0, 1.1)
        ax_p.set_title("2. Softmax: probabilities (add up to 100%)", fontsize=12)
        for ax in (ax_s, ax_p):
            ax.set_xticks(DIGITS)
            ax.set_xlabel("digit")
            _clean(ax)

        p = np.linspace(0.005, 1, 300)
        ax_l.plot(p, -np.log(p), color=GREY, lw=2)
        ax_l.plot(probs[target], loss, "o", ms=12, color=PRIMARY if guess == target else RED)
        ax_l.set_xlabel(f"probability given to the correct digit ({target})")
        ax_l.set_ylabel("loss")
        ax_l.set_ylim(0, 5.5)
        ax_l.set_title(f"3. Loss = −ln({probs[target]:.2f}) = {loss:.2f}", fontsize=12)
        _clean(ax_l)
        verdict = "correct" if guess == target else "wrong"
        fig.suptitle(f"The model predicts {guess} ({verdict}). Lower loss = better. "
                     "nn.CrossEntropyLoss does steps 2 and 3 for you.", fontsize=12)
        plt.tight_layout()
        plt.show()

    controls = {"target": target, **{f"s{d}": s for d, s in zip(DIGITS, scores)}}
    ui = widgets.VBox([widgets.HTML("<b>Drag the scores up and down:</b>"), widgets.HBox(scores), target])
    _show(controls, draw, ui)


def explore_gradient_descent():
    """Gradient descent on a simple curve: how the learning rate changes each step."""
    rate = _float_slider("Learning rate (step size)", 0.3, 0.05, 2.2, 0.05, width="420px")
    steps = _int_slider("Number of steps", 6, 0, 20)
    start = _float_slider("Starting weight", -2.0, -3.0, 9.0, 0.5, width="420px", fmt=".1f")

    def loss(w):
        return 0.5 * (w - 3) ** 2

    def draw(rate, steps, start):
        path = [start]
        for _ in range(steps):
            w = path[-1]
            path.append(w - rate * (w - 3))   # gradient of the loss is (w - 3)
            if abs(path[-1]) > 1e3:
                break
        path = np.array(path)

        fig, (ax, ax_hist) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
        ws = np.linspace(-4, 10, 300)
        ax.plot(ws, loss(ws), color=GREY, lw=2)
        ax.plot(3, 0, "*", ms=16, color=PRIMARY_DARK)
        ax.annotate("lowest loss", (3, 0), (3.4, 2), fontsize=10, color=PRIMARY_DARK,
                    arrowprops=dict(arrowstyle="-", color=PRIMARY_DARK))
        visible = [(w, loss(w)) for w in path if -4 <= w <= 10]
        for (w0, l0), (w1, l1) in zip(visible[:-1], visible[1:]):
            _arrow(ax, (w0, l0), (w1, l1), color=PRIMARY, lw=1.6)
        if visible:
            ax.scatter(*zip(*visible), color=PRIMARY, edgecolor=TEXT, zorder=3, s=45)
            ax.plot(*visible[0], "o", ms=11, color=TEXT)
        ax.set_xlim(-4, 10)
        ax.set_ylim(-1, 26)
        ax.set_xlabel("weight w")
        ax.set_ylabel("loss")
        _clean(ax)

        if rate < 0.98:
            status = "Each step moves closer to the lowest loss."
        elif rate <= 1.02:
            status = "One step lands exactly on the minimum (only on this simple curve)."
        elif rate < 1.98:
            status = "Steps overshoot and bounce from side to side, but still get closer."
        elif rate <= 2.02:
            status = "Steps bounce between the same two points forever."
        else:
            status = "Steps are too big: each one overshoots further. Training diverges."
        ax.set_title(f"new w = w − learning rate × slope\n{status}", fontsize=12)

        ax_hist.plot(range(len(path)), loss(path), "o-", color=PRIMARY)
        ax_hist.set_xlabel("step")
        ax_hist.set_ylabel("loss")
        ax_hist.set_ylim(0, max(loss(path).max() * 1.15, 1))
        ax_hist.set_title("Loss after each step", fontsize=12)
        _clean(ax_hist)
        plt.tight_layout()
        plt.show()

    _show({"rate": rate, "steps": steps, "start": start}, draw, widgets.VBox([rate, steps, start]))


_TRAIN_CODE = [
    "def train():",
    "    loss = 0",
    "    accuracy = 0",
    "",
    "    model.train()",
    "    for x, y in train_loader:",
    "        x, y = x.to(device), y.to(device)",
    "        output = model(x)",
    "        optimizer.zero_grad()",
    "        batch_loss = loss_function(output, y)",
    "        batch_loss.backward()",
    "        optimizer.step()",
    "",
    "        loss += batch_loss.item()",
    "        accuracy += get_batch_accuracy(output, y, train_N)",
    "    print('Train - Loss: ... Accuracy: ...')",
]

_TRAIN_STEPS = [
    ("Get a batch", [5, 6], "Take the next batch of images (x) and their labels (y) and move them "
                            "to the device (GPU or CPU)."),
    ("Predict", [7], "Forward pass: the model turns every image in the batch into 10 scores."),
    ("Clear gradients", [8], "Wipe the gradients of the previous batch. backward() adds new gradients to "
                             "the old ones instead of replacing them, so without this they would pile up."),
    ("Grade", [9], "The loss function compares the scores with the correct labels. "
                   "One number: lower is better."),
    ("Find direction", [10], "Backward pass: for every weight, work out whether increasing or decreasing "
                             "it would lower the loss (the gradient)."),
    ("Update weights", [11], "The optimizer moves every weight a small step in that direction."),
    ("Keep score", [13, 14], "Add this batch's loss and accuracy to the totals printed at the end."),
]


def show_training_loop(batches_per_epoch=None):
    """Step through what happens to every batch inside train()."""
    step = _int_slider("Step", 1, 1, len(_TRAIN_STEPS), width="320px")
    play = widgets.Play(value=1, min=1, max=len(_TRAIN_STEPS), interval=1800, description="Play")
    widgets.jslink((play, "value"), (step, "value"))

    def draw(step):
        current = step - 1
        fig, (ax, ax_code) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        count = len(_TRAIN_STEPS)
        angles = np.pi / 2 - 2 * np.pi * np.arange(count) / count
        points = [(1.4 * np.cos(a), 1.15 * np.sin(a)) for a in angles]
        for k in range(count):
            _arrow(ax, points[k], points[(k + 1) % count], color=LIGHT_GREY, lw=2.5, rad=-0.25)
        for k, ((x, y), (name, _, _)) in enumerate(zip(points, _TRAIN_STEPS)):
            active = k == current
            _box(ax, x - 0.5, y - 0.19, 1.0, 0.38, f"{k + 1}. {name}",
                 fc=PRIMARY if active else PRIMARY_LIGHT, color="white" if active else TEXT,
                 size=9.5, weight="bold" if active else "normal")
        repeat = ("Repeat for every batch" if batches_per_epoch is None
                  else f"Repeat for all\n{batches_per_epoch:,} batches")
        ax.text(0, 0.08, repeat, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(0, -0.28, "= 1 epoch", ha="center", va="center", fontsize=10, color=GREY)

        ax_code.axis("off")
        ax_code.set_xlim(0, 1)
        ax_code.set_ylim(0, 1)
        top, line_h = 0.97, 0.046
        for i, line in enumerate(_TRAIN_CODE):
            y = top - i * line_h
            if i in _TRAIN_STEPS[current][1]:
                ax_code.add_patch(Rectangle((0, y - line_h * 0.8), 1, line_h, fc=PRIMARY_LIGHT,
                                            ec=PRIMARY, lw=1))
            ax_code.text(0.01, y - line_h * 0.3, line, family="monospace", fontsize=9.5, va="center")
        ax_code.text(0, top - len(_TRAIN_CODE) * line_h - 0.04,
                     f"Step {step}: {_TRAIN_STEPS[current][0]}", fontsize=12, weight="bold",
                     color=PRIMARY_DARK, va="top")
        ax_code.text(0, top - len(_TRAIN_CODE) * line_h - 0.11, _TRAIN_STEPS[current][2],
                     fontsize=11, va="top", wrap=True)
        plt.tight_layout()
        plt.show()

    _show({"step": step}, draw, widgets.HBox([play, step]))


_STEP_X = 3.0      # input of the one-weight model
_STEP_Y = 10.0     # correct answer
_STEP_W0 = 2.0     # starting weight
_STEP_LR = 0.01    # learning rate of the SGD optimizer
_STEP_BEST = _STEP_Y / _STEP_X


def _step_loss(w):
    return (w * _STEP_X - _STEP_Y) ** 2


def explore_training_step():
    """Run the real zero_grad(), backward() and step() lines on a model with a single weight."""
    import torch

    wide = widgets.Layout(width="auto")
    buttons = {
        "forward": widgets.Button(description="1. Predict + grade (forward)", layout=wide),
        "zero_grad": widgets.Button(description="2. optimizer.zero_grad()", layout=wide),
        "backward": widgets.Button(description="3. batch_loss.backward()", layout=wide),
        "step": widgets.Button(description="4. optimizer.step()", layout=wide),
    }
    run_one = widgets.Button(description="Run 1 batch (1 → 4)", layout=wide, button_style="info")
    run_ten = widgets.Button(description="Run 10 batches", layout=wide, button_style="info")
    skip = widgets.Checkbox(value=False, description="Skip zero_grad() in Run", indent=False,
                            layout=widgets.Layout(width="200px", margin="0 0 0 12px"))
    reset = widgets.Button(description="Reset", layout=wide)
    message = widgets.HTML()
    out = widgets.Output()
    state = {}

    def start():
        w = torch.tensor(_STEP_W0, requires_grad=True)
        state.update(w=w, optimizer=torch.optim.SGD([w], lr=_STEP_LR), batch_loss=None,
                     history=[_STEP_W0], changed=set())
        return (f"A model with <b>one weight</b> <code>w = {_STEP_W0:g}</code>. It predicts <code>w × {_STEP_X:g}</code>, "
                f"the correct answer is <code>{_STEP_Y:g}</code>, and the loss is <code>(prediction − {_STEP_Y:g})²</code>. "
                "Each button runs that line with real PyTorch. Start with button 1.")

    def grad():
        g = state["w"].grad
        return None if g is None else g.item()

    def forward():
        output = state["w"] * _STEP_X
        state["batch_loss"] = (output - _STEP_Y) ** 2
        state["changed"] = {"batch_loss"}
        return (f"<b>Forward pass.</b> Prediction = {state['w'].item():.3f} × {_STEP_X:g} = {output.item():.2f}. "
                f"The correct answer is {_STEP_Y:g}, so the loss is ({output.item():.2f} − {_STEP_Y:g})² = "
                f"{state['batch_loss'].item():.2f}. PyTorch also recorded these calculations, "
                "so that <code>backward()</code> can go back through them.")

    def zero_grad():
        was = grad()
        state["optimizer"].zero_grad()
        state["changed"] = {"grad"}
        if was is None:
            return "<b>zero_grad():</b> <code>w.grad</code> was already empty. Nothing to wipe."
        return (f"<b>zero_grad()</b> wiped the old gradient ({was:.2f}). <code>w.grad</code> is now "
                "<code>None</code>, which means empty. The weight did not change.")

    def backward():
        state["changed"] = set()
        if state["batch_loss"] is None:
            return ("<b>Nothing to go back through.</b> <code>backward()</code> starts from the loss, "
                    "so run the forward pass (button 1) first.")
        was = grad()
        try:
            state["batch_loss"].backward()
        except RuntimeError as error:
            if "second time" not in str(error):
                return f"<b>PyTorch raised an error:</b> {error}"
            return ("<b>PyTorch raised an error:</b> <i>Trying to backward through the graph a second time</i>. "
                    "After <code>backward()</code>, PyTorch throws away its record of the forward pass to save "
                    "memory. Every batch needs a new forward pass (button 1).")
        now = grad()
        state["changed"] = {"grad"}
        if was:
            return (f"<b>backward() added</b> the new gradient ({now - was:.2f}) to the one already stored "
                    f"({was:.2f}), so <code>w.grad</code> = {now:.2f}. The next <code>step()</code> follows this "
                    "sum, old direction included. That is why <code>train()</code> calls <code>zero_grad()</code> "
                    "for every batch.")
        if now < 0:
            meaning = "Negative: making w bigger lowers the loss, so step() will increase w"
        elif now > 0:
            meaning = "Positive: making w bigger raises the loss, so step() will decrease w"
        else:
            meaning = "Zero: w is already at the best value, so step() will not move it"
        return (f"<b>backward()</b> worked out the gradient: <code>w.grad</code> = {now:.2f}. {meaning}. "
                f"The size, {abs(now):.2f}, is how steep the loss curve is at w = {state['w'].item():.3f}. "
                "The weight itself has not changed yet.")

    def step():
        g = grad()
        old = state["w"].item()
        state["optimizer"].step()
        if g is None:
            state["changed"] = set()
            return ("<b>step() did nothing:</b> <code>w.grad</code> is empty, so the optimizer has no direction "
                    "to follow. Run <code>backward()</code> (button 3) first.")
        new = state["w"].item()
        state["history"].append(new)
        state["changed"] = {"w"}
        return (f"<b>step()</b> moved the weight: w = {old:.3f} − {_STEP_LR:g} × ({g:.2f}) = {new:.3f}. "
                f"The prediction is now {new * _STEP_X:.2f} (correct answer: {_STEP_Y:g}). "
                "<code>step()</code> leaves <code>w.grad</code> as it is: wiping it is the job of "
                "<code>zero_grad()</code>.")

    def run(batches):
        before = state["w"].item()
        for _ in range(batches):
            forward()
            if not skip.value:
                zero_grad()
            backward()
            step()
        state["changed"] = {"w", "grad", "batch_loss"}
        after = state["w"].item()
        lines = "forward, backward, step, no zero_grad" if skip.value else "forward, zero_grad, backward, step"
        text = (f"<b>{batches} batch{'es' if batches > 1 else ''}</b> ({lines}): w went from "
                f"{before:.3f} to {after:.3f} (best value: {_STEP_BEST:.3f}), and the loss from "
                f"{_step_loss(before):.2f} to {_step_loss(after):.2f}.")
        if skip.value:
            text += (" Without <code>zero_grad()</code>, the old gradients pile up in <code>w.grad</code>. "
                     "The weight swings past the best value and back, and never settles. Follow the trail of dots.")
        elif batches > 1:
            text += " Each step is smaller than the one before, because the curve gets flatter near the bottom."
        return text

    def draw():
        w = state["w"].item()
        g = grad()
        loss_now = _step_loss(w)
        fig, (ax, ax_mem) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})

        ws = np.linspace(0.8, 5.9, 300)
        ax.plot(ws, _step_loss(ws), color=GREY, lw=2)
        ax.plot(_STEP_BEST, 0, "*", ms=16, color=PRIMARY_DARK, zorder=4)
        ax.annotate(f"best w = {_STEP_BEST:.3f}", (_STEP_BEST, 0), (_STEP_BEST, 6.5), ha="center",
                    fontsize=10, color=PRIMARY_DARK, arrowprops=dict(arrowstyle="-", color=PRIMARY_DARK))
        trail = [(h, _step_loss(h)) for h in state["history"][-30:]]
        for (w0, l0), (w1, l1) in zip(trail[:-1], trail[1:]):
            _arrow(ax, (w0, l0), (w1, l1), color=PRIMARY, lw=1.2)
        ax.scatter(*zip(*trail), color=PRIMARY_LIGHT, edgecolor=PRIMARY, s=30, zorder=3)

        title = ["Loss for every value of w (black dot = current w)"]
        if g is not None:
            d = 0.45
            true_slope = 2 * (w * _STEP_X - _STEP_Y) * _STEP_X
            if abs(g - true_slope) > 0.01:
                ax.plot([w - d, w + d], [loss_now - true_slope * d, loss_now + true_slope * d],
                        "--", color=GREY, lw=1.5)
            ax.plot([w - d, w + d], [loss_now - g * d, loss_now + g * d], color=PRIMARY, lw=2.5)
            target = w - _STEP_LR * g
            if abs(target - w) > 0.02:
                _arrow(ax, (w, loss_now), (np.clip(target, 0.8, 5.9), loss_now), color=PRIMARY_DARK, lw=2)
            title.append("Line = slope stored in w.grad. Arrow = where step() will move w.")
            if abs(g - true_slope) > 0.01:
                title.append("Dashed = the real slope here: w.grad no longer matches it.")
        ax.plot(w, loss_now, "o", ms=13, color=TEXT, zorder=5)
        ax.set_xlim(0.8, 5.9)
        ax.set_ylim(-1, 24)
        ax.set_xlabel("weight w")
        ax.set_ylabel("loss")
        ax.set_title("\n".join(title), fontsize=11)
        _clean(ax)

        ax_mem.set_xlim(0, 10)
        ax_mem.set_ylim(0, 10)
        ax_mem.axis("off")
        ax_mem.set_title("What PyTorch has in memory", fontsize=12, weight="bold")
        batch_loss = state["batch_loss"]
        rows = [
            ("w", f"{w:.3f}", "the weight: changed only by step()", "w"),
            ("w.grad", "None (empty)" if g is None else f"{g:.2f}",
             "written by backward(), wiped by zero_grad()", "grad"),
            ("batch_loss", "-" if batch_loss is None else f"{batch_loss.item():.2f}",
             "written by the forward pass", "batch_loss"),
        ]
        for k, (name, value, note, key) in enumerate(rows):
            y = 6.9 - 3.15 * k
            active = key in state["changed"]
            _box(ax_mem, 0.2, y, 9.6, 2.6, "", fc=PRIMARY if active else PRIMARY_LIGHT)
            ax_mem.text(0.6, y + 1.65, f"{name} = {value}", fontsize=15, family="monospace",
                        weight="bold", color="white" if active else TEXT, va="center")
            ax_mem.text(0.6, y + 0.6, note, fontsize=10, color="white" if active else GREY, va="center")
        plt.tight_layout()
        plt.show()

    def show(text):
        message.value = (f"<div style='border-left: 4px solid {PRIMARY}; padding: 6px 12px; "
                         f"margin: 6px 0; max-width: 900px'>{text}</div>")
        with out:
            out.clear_output(wait=True)
            draw()

    def on_click(action):
        return lambda _: show(action())

    for name, action in [("forward", forward), ("zero_grad", zero_grad), ("backward", backward),
                         ("step", step)]:
        buttons[name].on_click(on_click(action))
    run_one.on_click(on_click(lambda: run(1)))
    run_ten.on_click(on_click(lambda: run(10)))
    reset.on_click(on_click(start))

    display(widgets.VBox([
        widgets.HBox(list(buttons.values())),
        widgets.HBox([run_one, run_ten, skip, reset]),
        message,
        out,
    ]))
    show(start())


def explore_predictions(model, valid_set, device, batch_size=1000):
    """Browse the trained model's predictions on the validation images."""
    import torch
    from torch.utils.data import DataLoader

    net = getattr(model, "_orig_mod", model)  # skip torch.compile for this quick pass
    net.eval()
    batches = []
    with torch.no_grad():
        for x, _ in DataLoader(valid_set, batch_size=batch_size):
            batches.append(torch.softmax(net(x.to(device)), dim=1).cpu())
    probs = torch.cat(batches).numpy()
    preds = probs.argmax(axis=1)
    labels = np.asarray(valid_set.targets)
    mistakes = np.flatnonzero(preds != labels)
    accuracy = 1 - len(mistakes) / len(labels)

    only_mistakes = widgets.Checkbox(value=False, description="Only show mistakes")
    index = _int_slider("Image", 0, 0, len(labels) - 1, width="560px")

    def on_toggle(change):
        index.value = 0
        index.max = max(len(mistakes) - 1, 0) if change["new"] else len(labels) - 1

    only_mistakes.observe(on_toggle, names="value")

    def draw(only_mistakes, index):
        if only_mistakes and len(mistakes) == 0:
            print("No mistakes on the validation set!")
            return
        i = int(mistakes[min(index, len(mistakes) - 1)]) if only_mistakes else min(index, len(labels) - 1)
        label, guess = int(labels[i]), int(preds[i])
        right = label == guess

        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 2]})
        ax_img.imshow(_grid(valid_set.data[i]), cmap="gray", vmin=0, vmax=1)
        ax_img.set_title(f"Validation image #{i:,}\ncorrect: {label}   predicted: {guess}",
                         fontsize=12, color=PRIMARY_DARK if right else RED)
        ax_img.axis("off")
        colors = [LIGHT_GREY] * 10
        colors[guess] = PRIMARY if right else RED
        bars = ax_bar.bar(DIGITS, probs[i], color=colors)
        bars[label].set_edgecolor(PRIMARY_DARK)
        bars[label].set_linewidth(2.5)
        ax_bar.set_xticks(DIGITS)
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_xlabel("digit")
        ax_bar.set_ylabel("probability")
        ax_bar.set_title(f"How sure the model is (accuracy on all {len(labels):,} images: "
                         f"{accuracy:.2%}, {len(mistakes)} mistakes)", fontsize=11)
        _clean(ax_bar)
        plt.tight_layout()
        plt.show()

    _show({"only_mistakes": only_mistakes, "index": index}, draw, widgets.VBox([only_mistakes, index]))


# ---------------------------------------------------------------------------
# Check your understanding: certification-style questions
# ---------------------------------------------------------------------------

QUESTIONS = [
    dict(topic="Core ML and AI knowledge",
         question="A bank wants to read handwritten digits on cheques. Hand-written rules for every handwriting "
                  "style have failed. Why is deep learning a better fit for this task?",
         options=["It runs faster than traditional code on any hardware",
                  "It learns the rules from labelled examples instead of a person writing them",
                  "It does not need any data to get started",
                  "It guarantees 100% accuracy on new cheques"],
         answer=1,
         explanation="Deep learning learns the patterns from many examples and their correct answers. It still "
                     "needs data, and it is never guaranteed to be 100% accurate."),
    dict(topic="Data analysis",
         question="An MNIST image is converted with ToImage() followed by ToDtype(torch.float32, scale=True). "
                  "What are the shape and the value range of the resulting tensor?",
         options=["(28, 28), values 0 to 255",
                  "(1, 28, 28), values 0.0 to 1.0",
                  "(28, 28, 1), values 0.0 to 1.0",
                  "(784,), values 0 to 255"],
         answer=1,
         explanation="PyTorch images are channels-first (C × H × W). MNIST has 1 channel, and scale=True "
                     "divides the 0-255 pixel values by 255."),
    dict(topic="Experimentation",
         question="A colleague reports 99.9% accuracy, measured on the same images the model was trained on. "
                  "What is the main problem with this result?",
         options=["The batch size was too small",
                  "It does not show how well the model works on images it has never seen",
                  "Accuracy can only be measured on a GPU",
                  "Accuracy cannot be computed on training data"],
         answer=1,
         explanation="A model can memorise its training images. Only data it never trained on, like the "
                     "validation set, shows how well it generalises."),
    dict(topic="Experimentation",
         question="Why does the training DataLoader use shuffle=True while the validation DataLoader does not?",
         options=["Shuffling the validation data would change the validation accuracy",
                  "Shuffled training batches mix all classes; order does not change the validation score",
                  "Validation data must be sorted by label",
                  "Shuffling only works for the training split"],
         answer=1,
         explanation="Weights are updated after every training batch, so the order matters. During validation "
                     "no weights change, so the order of the images does not affect the result."),
    dict(topic="Core ML and AI knowledge",
         question="A batch of images with shape (32, 1, 28, 28) is passed through nn.Flatten(). "
                  "What is the output shape?",
         options=["(32, 784)", "(25088,)", "(32, 28, 28)", "(1, 784)"],
         answer=0,
         explanation="nn.Flatten() keeps the first (batch) dimension and joins the rest: 1 × 28 × 28 = 784 "
                     "values per image."),
    dict(topic="Core ML and AI knowledge",
         question="How many trainable parameters does nn.Linear(784, 512) have?",
         options=["401,408 (784 × 512)", "401,920 (784 × 512 + 512)", "1,296 (784 + 512)", "512"],
         answer=1,
         explanation="Each of the 512 neurons has one weight per input (784) and one bias: "
                     "784 × 512 weights + 512 biases."),
    dict(topic="Core ML and AI knowledge",
         question="Why is an activation function such as ReLU placed between two Linear layers?",
         options=["It makes the GPU run faster",
                  "Without it, two Linear layers equal one Linear layer and learn only straight-line patterns",
                  "It scales the inputs to the range 0.0 to 1.0",
                  "It reduces the number of parameters in the model"],
         answer=1,
         explanation="A Linear layer followed by another Linear layer is still one linear function. The "
                     "non-linear activation is what lets the network learn complex patterns."),
    dict(topic="Core ML and AI knowledge",
         question="The output layer has 10 neurons and no activation, and the model is trained with "
                  "nn.CrossEntropyLoss. Why is there no activation on the output layer?",
         options=["CrossEntropyLoss applies softmax itself and expects the raw scores (logits)",
                  "Output layers can never have an activation function",
                  "ReLU on the output would make training faster but less accurate",
                  "The 10 output scores must always be negative"],
         answer=0,
         explanation="nn.CrossEntropyLoss combines softmax and the negative log probability. It expects "
                     "raw scores, so adding softmax or ReLU before it would distort the loss."),
    dict(topic="Software development",
         question="In the training loop, what is the purpose of optimizer.zero_grad()?",
         options=["It sets all the weights to zero",
                  "It clears the previous batch's gradients, because PyTorch adds new gradients to old ones",
                  "It resets the learning rate to its initial value",
                  "It switches the model to evaluation mode"],
         answer=1,
         explanation="backward() accumulates (adds) gradients. Without zero_grad(), every update would also "
                     "include the gradients of all previous batches."),
    dict(topic="Experimentation",
         question="Over several epochs, training accuracy keeps rising while validation accuracy stops "
                  "improving and starts to drop. What is the most likely cause?",
         options=["Underfitting: the model is too small",
                  "Overfitting: the model memorises the training images",
                  "The learning rate is too small",
                  "The validation data is not shuffled"],
         answer=1,
         explanation="A growing gap between training and validation accuracy is the classic sign of "
                     "overfitting. Common fixes are more data, data augmentation or stopping earlier."),
    dict(topic="Performance optimization",
         question="Training with a very large learning rate makes the loss jump up and down or grow. "
                  "What is happening?",
         options=["Each update overshoots the minimum of the loss",
                  "The model has run out of GPU memory",
                  "The gradients have all become zero",
                  "The DataLoader is returning the same batch every time"],
         answer=0,
         explanation="The learning rate sets the step size. Steps that are too big jump past the minimum, "
                     "so the loss oscillates or diverges."),
    dict(topic="Software development",
         question="Why does validate() call model.eval() and run inside torch.no_grad()?",
         options=["To update the weights faster",
                  "No weights change, so gradients are not needed; eval() sets layers to inference mode",
                  "To shuffle the validation data",
                  "To move the model from the GPU to the CPU"],
         answer=1,
         explanation="torch.no_grad() skips gradient tracking, which saves memory and time. model.eval() "
                     "switches layers such as dropout and batch normalisation to inference behaviour."),
    dict(topic="Software development",
         question="A developer deletes optimizer.zero_grad() from train(). The code still runs without errors. "
                  "What goes wrong?",
         options=["Nothing: backward() replaces the old gradients anyway",
                  "The gradients of earlier batches pile up in .grad, so the weights follow old directions",
                  "The weights stop changing, because .grad stays at zero",
                  "The loss can no longer be computed"],
         answer=1,
         explanation="backward() adds new gradients to the ones already stored in .grad. zero_grad() wipes "
                     "them before each backward pass, so every update uses only the current batch."),
]


def quiz():
    """Certification-style multiple-choice questions about this notebook."""
    from .quiz import run_quiz
    run_quiz(QUESTIONS, seed=3)


# ---------------------------------------------------------------------------
# Practice exercises: automatic checks
# ---------------------------------------------------------------------------

def _base(model):
    return getattr(model, "_orig_mod", model)


def _device_of(model):
    return next(model.parameters()).device


def accuracy(model, loader):
    """Fraction of correctly classified images in a DataLoader."""
    import torch
    net = _base(model)
    device = _device_of(net)
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = net(x.to(device)).argmax(dim=1).cpu()
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total


def check_parameter_count(n_params, model):
    from .quiz import failed, passed
    expected = sum(p.numel() for p in model.parameters())
    if n_params is None:
        failed("Replace None with your calculation.")
    elif int(n_params) == expected:
        passed(f"The model has {expected:,} trainable parameters (weights and biases).")
    elif int(n_params) == sum(p.numel() for p in model.parameters() if p.dim() > 1):
        failed("That counts only the weights. Don't forget the biases.")
    else:
        failed(f"You got {int(n_params):,}. Hint: every tensor in model.parameters() has a .numel() method.")


def check_batch_predictions(preds, batch_acc, model, valid_loader):
    import torch
    from .quiz import failed, passed
    if preds is None or batch_acc is None:
        failed("Replace both None values with your code.")
        return
    x, y = next(iter(valid_loader))
    net = _base(model)
    net.eval()
    with torch.no_grad():
        expected = net(x.to(_device_of(net))).argmax(dim=1).cpu()
    preds = torch.as_tensor(preds).detach().cpu().flatten()
    if preds.shape != expected.shape:
        failed(f"preds should have shape {tuple(expected.shape)} (one digit per image), "
               f"not {tuple(preds.shape)}. Hint: use argmax(dim=1).")
        return
    if (preds == expected).sum().item() < len(expected) - 1:
        failed("preds does not match the model's predictions. Did you use the first batch of valid_loader?")
        return
    true_acc = (expected == y).float().mean().item()
    batch_acc = float(batch_acc)
    if abs(batch_acc - true_acc) < 1e-4:
        passed(f"The model got {int(round(true_acc * len(y)))} of the {len(y)} images in this batch right.")
    elif abs(batch_acc - true_acc * 100) < 1e-2:
        failed("Close: give the accuracy as a fraction between 0 and 1, not a percentage.")
    else:
        failed(f"preds is right, but batch_acc should be {true_acc:.4f}. "
               "Hint: compare preds with y, then take the mean.")


def check_small_model(small_model):
    import torch
    import torch.nn as nn
    from .quiz import failed, passed
    if small_model is None:
        failed("Replace None with your model.")
        return
    try:
        out = small_model.to("cpu")(torch.zeros(2, 1, 28, 28))
    except Exception as error:  # noqa: BLE001 - show any error to the student
        failed(f"The model fails on a batch of images: {error}")
        return
    if tuple(out.shape) != (2, 10):
        failed(f"The output should have shape (batch, 10), but it has shape {tuple(out.shape)}.")
        return
    if not any(isinstance(layer, nn.ReLU) for layer in small_model.modules()):
        failed("Add a ReLU between the two Linear layers.")
        return
    count = sum(p.numel() for p in small_model.parameters())
    if count != 101_770:
        failed(f"The model has {count:,} parameters. Expected 101,770: "
               "Linear(784, 128) and Linear(128, 10).")
        return
    passed("101,770 parameters, about 15% of the original model.")


def check_train_one_epoch(train_one_epoch, train_set, device):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from .quiz import failed, passed
    torch.manual_seed(0)
    loader = DataLoader(Subset(train_set, range(2048)), batch_size=64, shuffle=True)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10)).to(device)
    optimizer = torch.optim.Adam(net.parameters())
    before = [p.detach().clone() for p in net.parameters()]
    try:
        first = train_one_epoch(net, loader, optimizer, nn.CrossEntropyLoss())
        second = train_one_epoch(net, loader, optimizer, nn.CrossEntropyLoss())
    except Exception as error:  # noqa: BLE001 - show any error to the student
        failed(f"Your function raised an error: {error}")
        return
    if first is None:
        failed("The function should return the average loss of the epoch.")
        return
    first, second = float(first), float(second)
    if all(torch.equal(b, a.detach()) for b, a in zip(before, net.parameters())):
        failed("The weights did not change. Did you call loss.backward() and optimizer.step()?")
    elif first > 10:
        failed(f"The returned loss ({first:.1f}) looks like a sum. Return the average loss per batch.")
    elif second >= first:
        failed("The loss did not go down between two epochs. Did you call optimizer.zero_grad() each batch?")
    else:
        passed(f"The loss went down from {first:.3f} to {second:.3f}.")


def check_per_digit_accuracy(per_digit_acc, model, valid_loader):
    import torch
    from .quiz import failed, passed
    if per_digit_acc is None:
        failed("Replace None with your list of 10 accuracies.")
        return
    net = _base(model)
    net.eval()
    correct, total = torch.zeros(10), torch.zeros(10)
    with torch.no_grad():
        for x, y in valid_loader:
            preds = net(x.to(_device_of(net))).argmax(dim=1).cpu()
            for digit in range(10):
                mask = y == digit
                total[digit] += mask.sum()
                correct[digit] += (preds[mask] == digit).sum()
    expected = (correct / total).tolist()
    values = [float(v) for v in per_digit_acc]
    if len(values) != 10:
        failed(f"Expected 10 values (one per digit), got {len(values)}.")
    elif all(abs(a - b) < 2e-3 for a, b in zip(values, expected)):
        worst = int(torch.tensor(expected).argmin())
        passed(f"The hardest digit for this model is {worst} ({expected[worst]:.1%} correct).")
    else:
        failed("Some values are off. For each digit d: (predictions that are d among images labelled d) "
               "divided by (number of images labelled d).")
