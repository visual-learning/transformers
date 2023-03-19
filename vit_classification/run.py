

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from trainer import Trainer
from vit import ViT
from matplotlib import pyplot as plt

# load cifar10 training data from torchvision
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_dataloader =  DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_dataloader =  DataLoader(test_dataset, batch_size=64, shuffle=True)

device = 'cuda'
transformer = ViT(
                patch_dim=8,
                d_model=256,
                d_ff=1024,
                num_heads=4,
                num_layers=4,
                num_patches=16,
                num_classes=10,
                device=device
            )
trainer = Trainer(transformer, train_dataloader, test_dataloader, learning_rate=1e-4, batch_size=64, print_every=1, num_epochs=100)
trainer.train()

# Plot the training loss, train accuracy and test accuracy
plt.plot(trainer.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss history')
plt.savefig('loss_out.png')

plt.clf()
plt.xlabel("Epoch")
plt.ylabel('Accuracy')
plt.title("Accuracy vs. iterations")
plt.plot(*trainer.train_accuracy_history, label='train')
plt.plot(*trainer.test_accuracy_history, label='test')
plt.legend(loc='upper left')
plt.savefig('acc_out.png')