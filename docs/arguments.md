## *Arguments* 
             
Arguments supported in the kclassify model as hyperparameters.
      
| Argument                | Description                                                                                                             | Default                  |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------|--------------------------|
| --base_model            | vgg16, vgg19, efficientnetB0, mobilenetv2                                                                               | efficientnetB0           |
| --batch_size            | (optional)  Batch size                                                                                                  | 32                       |
| --lr                    | (optional)  Learning rate                                                                                               | .01                      |
| --has_wandb             | (optional)  Logs to the wandb server                                                                                    | False                    |
| --preprocessor          | use the model preprocessor on the inputs                                                                                | False                    |
| --featurewise_normalize | use featurewise centering and std normalizing                                                                           | True                     |
| --train_stats           | (optional) configuration file with training image statistics; must exist when using the --featurewise_normalize option  | -                        |
| --epochs                | (optional)  Number of epochs to train                                                                                   | 1                        |
| --optimizer             | (optional)  adam, radam, ranger                                                                                         | adam                     |
| --loss                  | (optional)  Type of loss function for the gradients: for the gradients categorical_crossentropy, or categorical_focal_loss | categorical_crossentropy |
| --dropout               | (optional) Add a drop out layer                                                                                         | False                    |
| --horizontal_flip       | (optional) Add horizontal flip augmentation during training                                                             | False                    |
| --vertical_flip         | (optional) Add vertical flip augmentation during training                                                               | False                    |
| --early_stop            | (optional) Add early stopping                                                                                           | False                    |
| --rotation_range        | (optional) Apply rotation augmentation between 0-1 as percent of image size during training                             | 0.0                      |
| --augment_range         | (optional) Apply width, shift, and zoom augmentation during training 0-1 as percent of image size                       | 0.0                      |
| --shear_range           | (optional) Apply sheer augmentation during training 0-1 as percent of image size                                        | 0.0                      |
 