# CenterNet

## Trianing Procedur

1. Download the EfficientDet D0 Pre-trained model
2. Place it a `models` folder
3. Open `src/lib/models/networks/efv0.py`, and change the `line 109` with the path of folder you replace the pre-trained weight
4. Create a folder in a `data` folder based on the of the project `ex: vibrio` to place the image dataset
5. Make three folders in it (annotations, train, val). Place all of the image and xml file in the train or val dataset. It is assumed that you have already split the dataset
6. open `xml_coco.py` file at folder `utils`
7. Change the class dictionary at `line 12` variable `category_set2`
8. Change the path of the image and xml path that you place at `CenterNet/data/project/train/`. at line 206. Then, edit line 207 with `instance_train.json`. if you use val folder, then `instance_val.json`
9. Move the `instances_train.json` and `instances_val.json` file to `CenterNet/data/annotations` folder
10. open file `CenterNet/src/lib/datasets/coco.py`. Then, edit line 14 `num_classes`, line 23 `self.data_dir = os.path.join(opt.data_dir, 'project name')`, and line 40-41 `self.class_name = [
      '__background__', 'green' 'yellow']`

11. open `CenterNet/src/lib/opts.py` file. Then, edit line 338 at key 'num_classes'. Adjust the number based on your number of class

12. Open a terminal at directory `CenterNet/src`. Then, type this command
```
python main.py ctdet --exp_id project name --arch efv0_0 --batch_size 4 --lr 1e-4 --gpus 0
```
Then, the model will start training

13. If you want to continue the training or want to load specific model, type this command
```
python main.py ctdet --exp_id project-name --arch efv0_0 --batch_size 4 --lr 1e-4 --gpus 0 --load_model ../path/to/weight.pth
```

14. The weight model will be saved at `CenterNet/exp/ctdet/project-name`

## Validation Proceduce
1. If you want to test the performance of the model. Open a terminal at directory `CenterNet/src`. Then, type this command
```
python test.py ctdet --exp_id project-name  --arch efv0_0 --load_model ../path/to/weight.pth
```