<!-- @format -->

## DETECTION OF COVID-19 OR PNEMONIA USING X-RAY IMAGES

### Include your name and ID here:

    1. Victor Umesiobi Q2094871
    2.Jingqi Yuan P1179164
    3. Jieyun Zhang P1180504
    4.Dingding Yao P1180133
    5.Ruchen Liu (p1180236)

### Problem Statement

`In 2020, covid hit the world, and we have x-ray images of the chest region and we want to be able to easily detect covid and its difference from pnemonia`

### Dataset

`https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset/data`

### Decision

1.  What kind of model do we use to train and why
2.  Cleaning and preparation of data
3.  Differences and relationships between data (showing graphs) - (Zhang)
4.  Training of data (victor)
5.  Inference (show graphs and accuracy) (yuan)
6.  Integrate with open cv (liu & Yao)
7.  Pratical use (phone application uses our model) (liu & Yao)

`Accuracy: 88.98%
Precision: 89.55%
Recall: 88.98%
F1 Score: 89.02%
Confusion Matrix:
[[421  26   4]
 [ 39 418   7]
 [ 16  60 388]]`

## To Run the Project

#### To Run the Project on Mobile

1. Connect to ssh and start the server, by running the command in your terminal

```
ssh vicksemmanuel@4.227.170.232
```

put the password

```
password > 00000000hB$00000000hB$
```

2. Connect as an admin and go to folder, by running the following commands

```
sudo su
```

```
cd Detect_Covid_19_And_Pneumnonia
```

3. Then run some git commands to update the repo
   > Note: This step is to update the code to the last updated version (it is not neccessary)

```
git fetch
```

```
git pull origin main
```

4. Now Run the Api

```
cd api && python3 run.py
```

5. Install the android apk on your phone and scan x-ray images [Download X-ray Detector Android App](https://expo.dev/artifacts/eas/xkhSvmGsrMQyzcvhCLc3jc.apk)

#### To Run the Project without Mobile Application

> Watch the video we've preprared

[Video](https://www.loom.com/share/73d5cc3eefdf4950a2d980f27ea4fc20?sid=7b87fc1e-ad77-4eaf-8baf-252320615f56)
