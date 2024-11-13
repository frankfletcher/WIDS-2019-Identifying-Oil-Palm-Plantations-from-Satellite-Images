# WIDS 2019 Datathon

### Project and Data Source: <https://www.kaggle.com/competitions/widsdatathon2019/overview>

### Purpose

Oil Palm Plantations are responsible for a large amount of deforestation. The goal of this project is to develop a model that can identify oil palm plantations in satellite images. This model can be used to monitor deforestation and help enforce laws that protect the environment.

### Primary Libraries Used: PyTorch/Torchvision

### Results

* Using a pre-trained ResNet-50 model, we were able to achieve a 0.93 F1 score and an **ROC-AUC of 0.98** on the holdout set. (<https://github.com/frankfletcher/WIDS-2019-Identifying-Oil-Palm-Plantations-from-Satellite-Images/blob/main/resnet50%20new%20EXP007.ipynb>)
* We also built a pre-trained ResNet-18 model which can be trained and used on a CPU. This model achieved an F1 score of 0.88 and a **ROC-AUC of 0.959** on the holdout set. (<https://github.com/frankfletcher/WIDS-2019-Identifying-Oil-Palm-Plantations-from-Satellite-Images/blob/main/resnet18%20new%20EXP014.ipynb>)

<br/>

![Metrics Showing ROC AUC .98](images/EXP007.png)

### Data Notes

* Data not included in this repository.  Please see the source link above to download the data.
* Preprocessing and data wrangling are done in the `wid_config.py` file.
  * combine the training and test labels
  * clean file names and keep track of paths
    * remove years from file names
  * drop duplicates (duplicates were created when dropping the years from the file names)
  * drop rows with missing values
  * drop rows where the score is below 1.0
  * Over/Under Sampling

### Model Notes

* The model is a pre-trained ResNet-50 model with a custom head.
* Optimizer is AdamW with minimal weight decay
* Various transformations were used on the training set to augment the data
* We use a freeze-thaw approach to tune the model
* We replicate the base transformations used to create the original resnet50 weights.
* Train and Validation were done on a 80/20 stratified split
* Early Stopping was used to prevent overfitting
* The model was trained on a single RTX 3090 GPU

### Future Directions

* Use a larger model such as ResNet-101 or ResNet-152
* Use MixUp or CutMix for data augmentation
* Label Smoothing
* Use a different optimizer such as Ranger
* Use a different learning rate scheduler

### Directory Structure

```bash
.
├── data                         # Data files including images
│   ├── holdout.csv
│   ├── SampleSubmission.csv
│   ├── testlabels.csv
│   └── traininglabels.csv
├── images                       # Images for README
├── models                       # Saved models
├── Pipfile
├── pytorch_base.ipynb           # Base notebook for PyTorch
├── README.md
├── resnet18 new EXP014.ipynb    # Best Small Model (ResNet-18)
├── resnet50 new EXP006.ipynb
├── resnet50 new EXP007.ipynb    # Best Model (ResNet-50)
├── resnet50 new EXP012.ipynb
├── wid_config.py                # Configuration file with wrangling
└── wid_torch.py                 # PyTorch experiments configurator class    
```
