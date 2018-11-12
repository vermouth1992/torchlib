# Generative Models
### Architecture Comparison
![Alt text](fig/architecture.png?raw=true "architecture")

### [Auxilliary Classifier GAN](https://arxiv.org/abs/1610.09585)
ACGAN did a pretty good job on mnist dataset.
On cifar10, IS(5.20±0.20). Already applied spectral normalization. 
Still need some insights to improve the quality. Will try [self-attention](https://arxiv.org/abs/1805.08318) later.

![Alt text](gan/acgan/fig/acgan_mnist.png?raw=true "ACGAN MNIST")
![Alt text](gan/acgan/fig/acgan_cifar10.png?raw=true "ACGAN CIFAR10")

Testing Accuracy

| MNIST  | Cifar10 |
| :---:  | :---:   |
| 99.11% | 72.11%  |

### [Semi-supervised GAN](https://arxiv.org/abs/1606.01583)
Train a normal GAN plus a header for classification in discriminator.
#### MNIST
![Alt text](gan/sgan/fig/sgan_mnist.png?raw=true "SGAN MNIST")

Testing Accuracy by using only 100 labeled training data.

|SGAN   | ConvNet (Same as D) |
| :---: | :---:   | 
|86.63% | 10.28% (Almost random guessing)  |

### [Bidirection GAN](https://arxiv.org/abs/1605.09782) ([ALI](https://arxiv.org/abs/1606.00704))
Visualize the latent space and use the latent code to perform classification.

### [InfoGAN](https://arxiv.org/abs/1606.03657)
Unsupervised learning of latent variables by maximizing mutual information.
#### MNIST
Can learn digit rotation and width automatically.

![Alt text](gan/infogan/fig/infogan_mnist_rotation.png?raw=true "INFOGAN MNIST")
![Alt text](gan/infogan/fig/infogan_mnist_width.png?raw=true "INFOGAN MNIST")

### [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

![Alt text](gan/wgan/fig/wgan_mnist.png?raw=true "WGAN MNIST")
