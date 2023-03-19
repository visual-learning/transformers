
import torch
from utils import * 
from torch.utils.data import DataLoader
from trainer import Trainer
from transformer import TransformerDecoder
from matplotlib import pyplot as plt

set_all_seeds(42) ### DO NOT CHANGE THIS LINE
exp_name = 'case1'

train_dataset = CocoDataset(load_coco_data(max_train=1024), 'train')
train_dataloader =  DataLoader(train_dataset, batch_size=64)

val_dataset = CocoDataset(load_coco_data(max_val = 1024), 'val')
val_dataloader =  DataLoader(val_dataset, batch_size=64)


device = 'cuda'
transformer = TransformerDecoder(
          word_to_idx=train_dataset.data['word_to_idx'],
          idx_to_word = train_dataset.data['idx_to_word'],
          input_dim=train_dataset.data['train_features'].shape[1],
          embed_dim=256,
          num_heads=2,
          num_layers=2,
          max_length=30,
          device = device
        )

trainer = Trainer(transformer, train_dataloader, val_dataloader,
          num_epochs=100,
          learning_rate=1e-4,
          device = device
        )

trainer.train()

# Plot the training losses.
plt.plot(trainer.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
os.makedirs('plots', exist_ok=True)
plt.title('Training loss history')
plt.savefig('plots/' + exp_name + '_loss_out.png')


def vis_imgs(split):
    data = {'train': train_dataset.data, 'val': val_dataset.data}[split]
    loader = {'train': train_dataloader, 'val': val_dataloader}[split]
    num_imgs = 0 
    for batch in loader:
      features, gt_captions, idxs = batch
      urls = data["%s_urls" % split][idxs]
      
      gt_captions = decode_captions(gt_captions, transformer.idx_to_word)
      sample_captions = transformer.sample(features, max_length=30)
      sample_captions = decode_captions(sample_captions, transformer.idx_to_word)
      
      for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
          img = image_from_url(url)
          # Skip missing URLs.
          if img is not None: 
            plt.imshow(img)            
            plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.savefig('plots/' + exp_name + '_%s_%d.png' % (split, num_imgs))
            num_imgs += 1
            if num_imgs >= 5: break
      return 

vis_imgs('train')
vis_imgs('val')