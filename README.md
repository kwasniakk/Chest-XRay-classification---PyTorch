# Chest X-Ray classification with PyTorch
This is my take on [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) using PyTorch transfer learning.
Checkpoints are available here on my [Google Drive](https://drive.google.com/drive/folders/1rZ8FixPrRzBQi9OX_gAJWXVfXPKyqjLV?usp=sharing)

### Folder structure 
chest_xray/
- test
    - NORMAL
    - PNEUMONIA
- train
    - NORMAL
    - PNEUMONIA
- val
    - NORMAL
    - PNEUMONIA


### Model
I'm using a pretrained version of ResNet18 from [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html). It is fine-tuned on a NVIDIA GTX 1050 Ti GPU.
### Evaluation  
The model seems to be handling the data pretty well, as shown in confusion matrices and recall, the model is more likely to classify healthy lungs as pneumonia infected lungs as only 7 out of 390 pneumonia images in validation dataset have been incorrectly classified, and all pneumonia test images have been correctly classified.
![confusion matrix](https://i.imgur.com/Q7o45i4.png)
 However, there are serious signs of overfitting.
![plots](https://i.imgur.com/SdaS72O.png)
  This might indicate that the model needs more hyperparameter tuning or ResNet18 is far from being an optimal choice for this specific task and perhaps another model might do the trick just right. Another issue is lack of data, since we are only given around 6000 images to train and validate the model on.

---
### Contributors  
- Krzysztof Kwaśniak 
    - [![Foo](https://i.imgur.com/nQueDcg.png)](https://www.linkedin.com/in/kwasniak-krzysztof/)
