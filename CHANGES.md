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
- Set the optimizer's learning rate explicitly (`Adam(model.parameters(), lr=0.001)`, the
  default) and added a plain-language explanation of the learning rate.

## `02_asl.ipynb`

- Updated the text that referred to `ToTensor()`.

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

## Removed

- The NVIDIA Deep Learning Institute header image (`images/DLI_Header.png`).
- The original slide decks. They contain NVIDIA logos and branding and Microsoft 365 stock
  images that may not be redistributed. Each deck is added back once it has been rebuilt
  without that material.
- NVIDIA's `CODEOWNERS` file and NVIDIA-specific instructions (Brev launch links).

## Additions

New material written for this course (interactive diagrams, quizzes, exercises, new slides,
FAQ, CPU Docker image) is not NVIDIA's material. It is listed in [`LICENSE`](LICENSE).
