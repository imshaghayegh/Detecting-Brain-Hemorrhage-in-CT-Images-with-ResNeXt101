# Detecting Brain Hemorrhage in CT Images with ResNeXt101

**Bachelor’s Project Report**

**Author:** Shaghayegh Shoushtari  
**Supervisor:** Dr. Mahmoud Farrokhian

---

## Abstract

This project uses a trained image model to detect whether a head CT slice contains hemorrhage.

A slice is one cross-sectional image from a CT scan. The selected model is **ResNeXt101** with a small channel-attention module called **ECA (Efficient Channel Attention)**. Images are prepared in a consistent way and passed through the network. The report focuses on the model’s **Any** output, which indicates the presence of hemorrhage.

The test evaluation includes **74,950 slices**. It reports:

- **Accuracy:** 95.22%
- **Sensitivity:** 75.80%
- **F1 Score:** 81.71%

The results show useful detection performance, but the model still misses some positive slices.

---

## 1. What the Project Does

A CT scan contains many slices. Each slice shows a different cross-section of the head. In this project, the model looks at one slice at a time and produces a score for whether hemorrhage is present. It does not use neighboring slices to make that decision.

The task is **image classification**: assigning an image to a category. Here, the two outcomes of interest are:

- Hemorrhage present
- Hemorrhage absent

The model is not asked to:

- Draw the boundary of the bleeding
- Measure its volume
- Recommend treatment

The project uses **transfer learning**. This means starting with a model that has already learned useful image patterns rather than starting with random weights. Those learned patterns provide a starting point for adapting the network to CT images. They do not mean that the original pretrained model already understood hemorrhage.

The work is organized around three steps:

1. Prepare the images
2. Obtain the network’s prediction
3. Compare that prediction with the known label

The main question is how well the selected model detects positive slices. A second question is whether its high overall accuracy gives a complete picture of its behavior.

The scope contains one CNN. It does not include feature reduction, a separate tree classifier, a recurrent network, or a comparison of several architectures. **ResNeXt101** remains the backbone throughout the project.

---

## 2. Data and Image Preparation

### The Images and Their Labels

The images come from the **RSNA 2019 Intracranial Hemorrhage Detection** dataset.

For the **Any** label:

- `1` means hemorrhage is present
- `0` means hemorrhage is absent

The saved test evaluation contains the following slices:

| Test Group | Slices |
|---|---:|
| Hemorrhage absent | 64,392 |
| Hemorrhage present | 10,558 |
| **Total** | **74,950** |

Most test slices are negative. This matters because a program that always predicts no hemorrhage would already be correct on about **85.91%** of this test set.

For this reason, it is important to measure how many positive slices the model finds, as well as how often it is correct overall.

The data have separate:

- Training partitions
- Validation partitions
- Test partitions

Training data are used to learn the model’s weights. Validation data are used to check choices during development. Test data are used for the final assessment.

The supplied splitting code groups images by imaging series. Patient overlap should also be checked because one patient may have several series.

### Preparing a Slice for the Network

The images are stored as **DICOM** files. DICOM contains both the image and information needed to read its pixel values correctly.

The loader uses that information to convert the stored values to **Hounsfield Units**, the intensity scale used for CT images.

Next, the code creates three views of the same slice using different intensity ranges:

- Brain
- Subdural
- Soft tissue

This process is called **windowing**. It is similar to choosing different contrast settings so that different details are easier to see.

These are three views of one slice, not three separate scans.

The three views are stacked as the input image’s three channels. The image is resized to **256 × 256 pixels**. Its values are then normalized using the channel means and standard deviations specified in the code.

During training, small random image changes provide additional variation. Test images use fixed preparation without random changes.

Keeping the preparation consistent is important because a trained model may behave differently if the:

- Channel order
- Image size
- Intensity handling

are changed.

---

## 3. How the Model Works

### Learning Useful Image Patterns

**ResNeXt101** is a convolutional neural network.

A convolutional network uses learned filters to find patterns in an image. Earlier layers can describe simpler patterns such as edges and changes in intensity. Later layers combine this information into more complex features.

The selected implementation uses the pretrained:

`resnext101_32x8d_wsl`

backbone.

It also includes **ECA**, a small module that adjusts the importance of feature channels. These internal channels describe patterns learned by the network and are different from the three channels of the input image.

The ECA output is added to the original features before the final classification stage.

The network then averages the feature maps to produce a summary of the slice: a vector of **2048 numbers**. These numbers are learned image features, not 2048 pixels or disease categories.

A final linear layer turns them into six raw scores.

### Using the `Any` Output

The existing model has five outputs for hemorrhage subtypes and one output called **Any**.

This report uses **Any** to answer the simpler question:

> Is hemorrhage present?

The other outputs remain in place so that the model is compatible with its existing weights. This is not a newly trained model with only one output.

A raw score is called a **logit**. A sigmoid function converts it to a value between zero and one.

A threshold then turns the score into a yes-or-no decision.

The exact threshold matters because changing it can change:

- The number of detected cases
- The number of false alarms

### Training and Running the Implementation

The code uses:

- Python
- PyTorch

Visible training settings include:

- **Batch size:** 32
- **Optimizer:** Adam
- **Learning rate:** 0.00001
- **Loss function:** `BCEWithLogitsLoss`

The evaluation stage loads the trained ECA model checkpoint and applies it to the test set.

A checkpoint stores the learned model weights, while an epoch is one complete pass through the training data.

To reproduce the same evaluation, the corresponding:

- Checkpoint
- Dataset split
- Preprocessing settings

must be used.

---

## 4. Results and Conclusion

### Test Results

The test evaluation reports the following values for the **Any** output on **74,950 slices**:

| Measure | Value |
|---|---:|
| Accuracy | 95.22% |
| Precision | 88.62% |
| Sensitivity | 75.80% |
| F1 Score | 81.71% |
| AUC | 0.9689 |

A sensitivity of **75.80%** means roughly 76 out of every 100 positive slices are detected, while about 24 are missed.

Precision answers a different question: among 100 positive predictions, roughly 89 are correct.

Accuracy counts all correct decisions, while F1 combines precision and sensitivity.

The evaluation compares raw logits with **0.5**, equivalent to a sigmoid probability threshold of about **0.6225**.

Changing this decision rule requires a new evaluation.

AUC measures score ranking rather than the percentage of correct decisions.

### What the Results Tell Us

The selected CNN provides useful separation between positive and negative slices in the saved experiment.

However, high accuracy does not mean every hemorrhage is detected. The lower sensitivity is an important limitation.

The model also processes slices independently, and its performance on an independent hospital dataset has not been established.

For reproducible evaluation, the same:

- Checkpoint
- Dataset split
- Preprocessing settings

should be used.

Patient separation between data splits should also be checked.

**ResNeXt101 remains computationally demanding.**

---

## Project Files

- [`Shaghayegh_Shushtari_Bachelor_Project_Report.pdf`](./Shaghayegh_Shushtari_Bachelor_Project_Report.pdf)
- [`Shaghayegh_Shushtari_Bachelor_Project.ipynb`](./Shaghayegh_Shushtari_Bachelor_Project.ipynb)

---

## References

1. Rahimi, P. *Detection and Classification of Brain Hemorrhage in CT Images Using Deep Learning*. Shahid Chamran University of Ahvaz. The broader study uses RSNA Intracranial Hemorrhage Detection data, 2019.
2. Project implementation: `Shaghayegh_Shushtari_Bachelor_Project.ipynb`
3. Xie, S., et al. *Aggregated Residual Transformations for Deep Neural Networks*. CVPR, 2017.
4. Wang, Q., et al. *ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks*. CVPR, 2020.

---

## Full Report

The complete English bachelor project report is available in this repository:

[**View the full PDF report**](./Shaghayegh_Shushtari_Bachelor_Project_Report.pdf)
