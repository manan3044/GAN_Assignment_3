# GAN_Assignment_3

### Deep Convolutional Generative Adversarial Network (DCGAN) with CelebA Dataset

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch to generate realistic face images from the CelebA dataset.

## Files in Repository 
There are 2 files included in the repository both belonging to same assignment:
1. **Assignment_3**: This Python notebook contains the implementation of a DCGAN model, including the generator and discriminator, trained on the CelebA dataset for 10 epochs.
2. **Assignment_3_5epochs**: This Python notebook contains the implementation of a DCGAN model, including the generator and discriminator. This is the same code the only difference is the DCGAN is trained on the CelebA dataset for only 5 epochs.

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

![image](https://github.com/user-attachments/assets/3ee0971c-ac43-4dc6-86ae-08e61801f376)


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
![image](https://github.com/user-attachments/assets/2502c621-108f-4500-8ff4-c5059e8f765f)

## Results
1. **Discriminator Loss (Loss_D)**  
- Starts high (~1.58) and fluctuates before stabilizing around 0.4 - 0.7.
- Higher Loss_D (~2.47) indicates the Discriminator is struggling to distinguish real and fake samples.
- Lower Loss_D (~0.3 - 0.5) suggests the Discriminator is getting better at classification but should not overpower the Generator.

2. **Generator Loss (Loss_G)**  
- Initially starts high (~2.34) and gradually decreases, suggesting that the Generator is improving.
- A very low Generator loss (~0.1 - 0.3) might indicate mode collapse, where the Generator produces limited variation.

3. **Training Stability**  
- Ideal training results show Loss_D and Loss_G converging while maintaining some balance.
- Large fluctuations may indicate instability and require hyperparameter tuning.


## References
- DCGAN Paper: `https://arxiv.org/abs/1511.06434`
- PyTorch DCGAN Tutorial: `https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html`

