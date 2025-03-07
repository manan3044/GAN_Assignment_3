# GAN_Assignment_3

### Deep Convolutional Generative Adversarial Network (DCGAN) with CelebA Dataset

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch to generate realistic face images from the CelebA dataset.

## Dataset Preprocessing Steps

1. **Download the CelebA Dataset**
   - Ensure you have the CelebA dataset downloaded.
   - The dataset consists of thousands of face images used for training the GAN.

2. **Load and Preprocess the Dataset**
   - Convert images to RGB format.
   - Resize images to the required size (e.g., 64x64 pixels).
   - Normalize pixel values to the range [-1, 1] for better GAN training.
   - Convert images into PyTorch tensors.

3. **Create DataLoader**
   - Use PyTorch's `DataLoader` to batch and shuffle the dataset for efficient training.

## How to Train the Model

1. **Setup Environment**
   - Install dependencies: `torch`, `torchvision`, `numpy`, `matplotlib`.
   - Ensure GPU acceleration is enabled (if available) for faster training.

2. **Initialize the Models**
   - Define the Generator and Discriminator architectures.
   - Initialize model weights using a normal distribution.

3. **Train the Model**
   - Define hyperparameters such as batch size, learning rate, and number of epochs.
   - Use Binary Cross Entropy Loss (BCE Loss) for both Generator and Discriminator.
   - Use Adam optimizer for both models.
   - Train the Discriminator to distinguish real vs. fake images.
   - Train the Generator to generate realistic images from random noise.
   - Save model checkpoints for future use.

4. **Monitor Training Progress**
   - Save and visualize generated images at regular intervals to evaluate performance.
   - Track loss curves for both Generator and Discriminator.

## How to Test the Model

1. **Load the Trained Generator**
   - Load the saved model checkpoint of the Generator.

2. **Generate Images**
   - Provide random noise as input to the Generator.
   - Generate and visualize synthetic face images.

## Expected Outputs

- Initially, the generated images will appear noisy and unrealistic.
- As training progresses, the Generator improves, and the images start resembling real human faces.
- After sufficient epochs, the GAN produces high-quality face images that are visually convincing.

## Notes
- Training GANs can be computationally expensive. A powerful GPU is recommended.
- Mode collapse (where the Generator produces limited variety) may occur and can be mitigated with proper hyperparameter tuning.
- Experimenting with different architectures, activation functions, and optimizers may improve results.

## References
- `DCGAN Paper: https://arxiv.org/abs/1511.06434`
- PyTorch DCGAN Tutorial: `https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html`

