# Changes to NVIDIA's original material

"Deep Learning, Visually" is adapted from
[Fundamentals of Deep Learning](https://github.com/NVDLI/fundamentals-of-deep-learning)
by NVIDIA (Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES), as published at commit
`d7b4b36` of that repository. NVIDIA's material is used under the Apache License 2.0 (code)
and the Creative Commons Attribution 4.0 International license (content).

This file lists the changes made to NVIDIA's material, as required by those licenses.

## All notebooks

- Replaced the NVIDIA Deep Learning Institute header image with an attribution line, and
  removed the closing header image.
- Rewrote the license notice to credit NVIDIA, link to this file and state that the course
  is not affiliated with NVIDIA.
- Changed image and Google Colab links to point to this repository.

## `01_mnist.ipynb`

- Replaced the deprecated `ToTensor()` transform with
  `Compose([ToImage(), ToDtype(torch.float32, scale=True)])` and updated the text that
  explains it.
- Made the prediction cell device-agnostic (`x_0_tensor.to(device)` instead of `x_0_gpu`),
  so the notebook runs to the end without a GPU.

## `02_asl.ipynb`

- Updated the text that referred to `ToTensor()`.
- `train()` and `validate()` also return the average loss and the accuracy; the training loop
  stores them in `history` and numbers the epochs from 1.
- Added a note after the overfitting discussion: with the current version of the dataset,
  validation accuracy is usually almost as high as training accuracy, because most validation
  images have a near-identical image in the training set.

## Docker

- `course_content/Dockerfile`: renamed the working directory from `/dli` to `/course` and
  added a pointer to the new CPU image.
- `course_content/README.md`: updated paths, image names and setup instructions.

## `slides_01.pptx`

- New title, colour theme and fonts; removed all NVIDIA logos and brand backgrounds.
- Removed all Microsoft 365 stock images and icons, third-party logos and a screenshot of
  a third-party video; replaced them with new diagrams or with openly licensed photos.
- Removed the Microsoft 365 content notice slide.
- Replaced the copyright footer with an attribution footer, and added a Credits slide.
- Added four new slides (an image's path through the model, an image as numbers, one
  neuron, the training loop) and a closing slide.

## `slides_02.pptx`

- New title, colour theme and fonts; removed all NVIDIA logos and brand backgrounds.
- Removed the Microsoft 365 content notice slide, stock images and icons, and screenshots of
  third-party websites; replaced them with new diagrams and with the same code as native text.
- Recoloured NVIDIA's loss-surface and activation-function plots.
- Rebuilt the ASL alphabet slide from the public-domain `asl.png`.
- Replaced the copyright footer with an attribution footer, and added a Credits slide.
- Added three new slides (from a CSV row to a tensor, Dataset and DataLoader, learning curves
  and overfitting) and a closing slide.

## Removed

- The NVIDIA Deep Learning Institute header image (`images/DLI_Header.png`).
- The original slide decks. They contain NVIDIA logos and branding and Microsoft 365 stock
  images that may not be redistributed. Each deck is added back once it has been rebuilt
  without that material.
- NVIDIA's `CODEOWNERS` file and NVIDIA-specific instructions (Brev launch links).

## Additions

New material written for this course (interactive diagrams, quizzes, exercises, new slides,
FAQ, CPU Docker image) is not NVIDIA's material. It is listed in [`LICENSE`](LICENSE).
