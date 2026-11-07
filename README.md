# Deep Learning, Visually

Learn deep learning with PyTorch by building, training and *seeing* neural networks.
Every lesson is a Jupyter notebook you can run in your browser, with interactive
diagrams you control with sliders and buttons, quizzes in the style of certification
exams, and coding exercises that check your answers.

> Adapted from [Fundamentals of Deep Learning](https://github.com/NVDLI/fundamentals-of-deep-learning)
> by NVIDIA, used under the Apache License 2.0 and CC BY 4.0. This course is not affiliated with,
> sponsored or endorsed by NVIDIA. See [`LICENSE`](LICENSE) and [`CHANGES.md`](CHANGES.md).

## What you will learn

- Build and train neural networks with PyTorch on real image datasets
- Apply convolutional neural networks (CNNs) to image classification
- Use data augmentation and transfer learning to get more out of limited data
- Recognise and fix overfitting
- Use a trained model to make predictions on new data
- Use a pre-trained language model to answer questions about a text

## What makes this course different

- **Interactive diagrams.** Move a slider and watch a neuron, a loss function or the
  training loop react. Every key idea has a picture you can play with.
- **Check your understanding.** Each lesson ends with multiple-choice questions in the
  style of professional certification exams, with an explanation for every answer.
- **Practice exercises.** Write the code yourself; the notebook checks it and gives hints.
- **Runs anywhere.** Google Colab, Docker with or without a GPU (including Apple Silicon
  Macs), or a local Python environment.

## Lessons

| # | Lesson | Notebook | Open in Colab | Interactive version |
|---|---|---|---|---|
| 00 | Working in JupyterLab | [00_jupyterlab.ipynb](course_content/tutorials/00_jupyterlab.ipynb) | | — |
| 01 | Image classification with MNIST | [01_mnist.ipynb](course_content/tutorials/01_mnist.ipynb) | [Open](https://colab.research.google.com/github/OblackatO/deep-learning-visually/blob/main/course_content/tutorials/01_mnist.ipynb) | ✓ Diagrams, quiz, exercises |
| 02 | How a neural network trains | [02_asl.ipynb](course_content/tutorials/02_asl.ipynb) | [Open](https://colab.research.google.com/github/OblackatO/deep-learning-visually/blob/main/course_content/tutorials/02_asl.ipynb) | Planned |
| 03 | Convolutional neural networks | [03_asl_cnn.ipynb](course_content/tutorials/03_asl_cnn.ipynb) | [Open](https://colab.research.google.com/github/OblackatO/deep-learning-visually/blob/main/course_content/tutorials/03_asl_cnn.ipynb) | Planned |
| 04a | Data augmentation | [04a_asl_augmentation.ipynb](course_content/tutorials/04a_asl_augmentation.ipynb) | [Open](https://colab.research.google.com/github/OblackatO/deep-learning-visually/blob/main/course_content/tutorials/04a_asl_augmentation.ipynb) | Planned |
| 04b | Deploying a model | [04b_asl_predictions.ipynb](course_content/tutorials/04b_asl_predictions.ipynb) | [Open](https://colab.research.google.com/github/OblackatO/deep-learning-visually/blob/main/course_content/tutorials/04b_asl_predictions.ipynb) | Planned |
| 05a | Pre-trained models | [05a_doggy_door.ipynb](course_content/tutorials/05a_doggy_door.ipynb) | [Open](https://colab.research.google.com/github/OblackatO/deep-learning-visually/blob/main/course_content/tutorials/05a_doggy_door.ipynb) | Planned |
| 05b | Transfer learning | [05b_corgi_door.ipynb](course_content/tutorials/05b_corgi_door.ipynb) | [Open](https://colab.research.google.com/github/OblackatO/deep-learning-visually/blob/main/course_content/tutorials/05b_corgi_door.ipynb) | Planned |
| 06 | Natural language processing | [06_nlp.ipynb](course_content/tutorials/06_nlp.ipynb) | [Open](https://colab.research.google.com/github/OblackatO/deep-learning-visually/blob/main/course_content/tutorials/06_nlp.ipynb) | Planned |

Slides: [Part 1 · An introduction to deep learning](course_content/slides/slides_01.pptx). The other decks are being redesigned and are added lesson by lesson.
Questions that come up while working through the lessons are answered in the
[FAQ](course_content/FAQ.md).

## Exam-prep pack

Want to go further? The **exam-prep pack** adds, for every lesson:

- **Mock exams**: timed, multiple-choice exams in the style of professional certification
  exams, with an explanation for every answer, plus a final exam for the whole course.
- **Projects with full solutions**: larger end-of-lesson projects on new datasets, with
  complete, commented solution notebooks.

**Price: €19** (launch price: €12). Payment is handled by Lemon Squeezy, which takes care
of VAT and sends you an invoice. *Coming soon.*

## Getting started

### Google Colab (no installation)

Click an **Open** link in the table above. In Colab, choose a GPU under
**Runtime → Change runtime type**, then run the **Setup for Google Colab** cell at the top
of the notebook.

### Docker

Run these commands from the `course_content/` directory, then open
<http://localhost:8888> and go to the `tutorials/` folder.

**Linux with an NVIDIA GPU** (requires the GPU driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
docker build -t dlv-app . && docker run -d -p 8888:8888 --name dlv-container --gpus all dlv-app
```

**macOS, or any machine without a GPU** (runs natively on Apple Silicon and x86-64):

```bash
docker build -f Dockerfile.cpu -t dlv-app-cpu . && docker run -d -p 8888:8888 --name dlv-container dlv-app-cpu
```

To keep your notebook changes on your computer, add
`-v "$PWD/tutorials:/course/tutorials"` to the `docker run` command.

> [!NOTE]
> On macOS, do not use the GPU `Dockerfile` or `--gpus all`. Docker Desktop cannot pass a
> GPU through to containers on macOS, and the GPU image runs under emulation on Apple
> Silicon, where notebook cells never finish. The CPU image is fine for the first lessons;
> for the later ones, a GPU (for example in Colab) is much faster.

### Local Python environment

Requires Python 3.10 or newer.

```bash
git clone https://github.com/OblackatO/deep-learning-visually.git
cd deep-learning-visually/course_content
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1
pip install -r environment/requirements.txt jupyterlab ipywidgets
jupyter lab tutorials
```

## License

- **NVIDIA's original material** keeps its licenses: Apache License 2.0 for code and
  CC BY 4.0 for content. The changes made to it are listed in [`CHANGES.md`](CHANGES.md).
- **Additions** (interactive diagrams, quizzes, exercises, new slides, FAQ and this README)
  are © 2026 Pedro Gomes, all rights reserved. You may use them for your own personal,
  non-commercial learning, but not share, publish or sell them. See
  [`LICENSES/LicenseRef-DLV-Personal-Use.txt`](LICENSES/LicenseRef-DLV-Personal-Use.txt).
- **Third-party images, data and models** keep their own licenses; see
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

NVIDIA names and logos are trademarks of NVIDIA and are not licensed.
