# Classification and Captioning Aircraft Damage Using Pretrained Models

## Project Overview
Aircraft damage detection is essential for maintaining the safety and longevity of aircraft[cite: 1]. Traditional manual inspection methods are time-consuming and prone to human error[cite: 1]. This project aims to automate the classification of aircraft damage into two categories: "dent" and "crack"[cite: 1]. To achieve this, the system utilizes feature extraction with a pre-trained VGG16 model to classify the damage from aircraft images[cite: 1]. Additionally, it uses a pre-trained Transformer model to generate captions and summaries for the images, providing a detailed description of the damage[cite: 1].

## Aim & Objectives
The goal of this project is to develop an automated model that accurately classifies aircraft damage from images, which is applicable in real-world damage detection within the aviation industry[cite: 1]. 
Key objectives include:
*   Using the VGG16 model for binary image classification[cite: 1].
*   Preparing and preprocessing image data for a machine learning task[cite: 1].
*   Utilizing custom Keras layers on top of the frozen base model[cite: 1].
*   Evaluating the model’s performance using appropriate metrics and visualizing model predictions on test data[cite: 1].
*   Generating captions and summaries of an image using the BLIP pretrained model[cite: 1].

## Dataset
The project utilizes the [Aircraft dataset](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar)[cite: 1]. The original source is the [Roboflow Aircraft Dataset](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk), provided by a Roboflow user under the CC BY 4 License[cite: 1]. The data is split into `train`, `valid`, and `test` directories containing `dent` and `crack` subfolders[cite: 1].

## Technical Stack & Dependencies
The following libraries are required to run the environment[cite: 1]:
*   `pandas==2.2.3`[cite: 1]
*   `tensorflow_cpu==2.17.1`[cite: 1]
*   `pillow==11.1.0`[cite: 1]
*   `matplotlib==3.9.2`[cite: 1]
*   `transformers==4.38.2`[cite: 1]
*   `torch==2.2.0+cpu`, `torchvision==0.17.0+cpu`, `torchaudio==2.2.0+cpu`[cite: 1]

## Methodology

### 1. Data Preprocessing
*   **Image Augmentation & Scaling:** Utilized Keras `ImageDataGenerator` to rescale pixel values (`1./255`)[cite: 1].
*   **Generators:** Configured `train_generator`, `valid_generator`, and `test_generator` to resize images to `224x224` target dimensions and process them in a batch size of 32[cite: 1]. The class mode was set to binary[cite: 1].

### 2. VGG16 Classification Model Architecture
*   **Base Model:** Loaded the pre-trained VGG16 model with `weights='imagenet'` and `include_top=False`, targeting an input shape of `(224, 224, 3)`[cite: 1]. The base VGG16 layers were frozen so their weights would not update during training[cite: 1].
*   **Custom Classifier Head:** The output from the last layer of the base model was flattened and passed through two fully connected `Dense` layers (512 units each, ReLU activation)[cite: 1]. To prevent overfitting, a `Dropout(0.3)` layer was added after each dense layer[cite: 1].
*   **Output Layer:** A final `Dense` layer with 1 unit and a `sigmoid` activation function was added to output the probability for binary classification[cite: 1].
*   **Compilation:** The model was compiled using the `Adam` optimizer with a learning rate of `0.0001`, `binary_crossentropy` loss, and tracking `accuracy` metrics[cite: 1].
*   **Training:** The model was trained for 5 epochs using the training generator and validated against the validation generator[cite: 1].

### 3. Image Captioning & Summarization
*   Implemented the Hugging Face BLIP pretrained model to process aircraft images and automatically generate descriptive text and summaries regarding the visible surface condition[cite: 1].

## Results & Evaluation
After 5 epochs of training, the model's performance was evaluated on the unseen test set[cite: 1]:
*   **Test Loss:** 0.6662[cite: 1]
*   **Test Accuracy:** 0.7500 (75.00%)[cite: 1]

Visualizations mapping true labels versus predicted labels are generated using `matplotlib` to verify performance[cite: 1].

## Installation & Usage
1. Clone this repository to your local machine.
2. Ensure you have Python installed, then install the required dependencies using `pip`:
   ```bash
   pip install pandas==2.2.3 tensorflow_cpu==2.17.1 pillow==11.1.0 matplotlib==3.9.2 transformers==4.38.2 torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu
