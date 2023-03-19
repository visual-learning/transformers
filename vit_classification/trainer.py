import numpy as np
import torch
from vit import ViT

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, learning_rate = 0.001, batch_size = 100, 
            num_epochs = 10, print_every = 10, save_every=10, verbose = True, device = 'cuda'):
      
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.save_every = save_every
        self.verbose = verbose 
        self.loss_history = []
        self.device = device
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=10, T_mult=1, eta_min=1e-6)
        self.test_accuracy_history = [[], []]
        self.train_accuracy_history = [[], []]

    def eval(self, dataloader):
        total_datapoints = 0
        correct_predictions = 0

        self.model.eval()

        for images, labels in dataloader:
            with torch.no_grad():
                logits = self.model(images.to(self.device))
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels.to(self.device)).item()
            total_datapoints += images.shape[0]

        self.model.train()

        return correct_predictions / total_datapoints

    def loss(self, predictions, labels):
        """
            Compute cross entropy loss between predictions and labels.
            Inputs:
                - predictions: PyTorch Tensor of shape (N, C) giving logits for each class
                - labels: PyTorch Tensor of shape (N,) giving labels for each input
        """

        
        # TODO - Compute cross entropy loss between predictions and labels. 
        loss = None
        

        return loss

    def train(self):
        for i in range(self.num_epochs):
            epoch_loss = 0
            num_batches = 0

            for images, labels in self.train_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)

                loss = self.loss(logits, labels)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                epoch_loss += loss.detach().cpu().numpy()
                num_batches += 1
                
            self.loss_history.append(epoch_loss/num_batches)
            self.scheduler.step()

            if self.verbose and i % self.print_every == 0:
                print( "(epoch %d / %d) loss: %f" % (i , self.num_epochs, self.loss_history[-1]))
                
                test_accuracy = self.eval(self.test_dataloader)
                self.test_accuracy_history[0].append(i)
                self.test_accuracy_history[1].append(test_accuracy)
                print( "Test accuracy: %f" % (test_accuracy))
                
                train_accuracy = self.eval(self.train_dataloader)
                self.train_accuracy_history[0].append(i)
                self.train_accuracy_history[1].append(train_accuracy)
                print( "Train accuracy: %f" % (train_accuracy))
            
            if i % self.save_every == 0:
                torch.save(self.model.state_dict(), 'checkpoints/model_{}.pth'.format(i))