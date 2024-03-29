# Exploring the Interactive Guidance for Unified and Effective Image Matting

This repo is the official PyTorch implementation of **Exploring the Interactive Guidance for Unified and Effective Image Matting**. Since this work is still under review, we only provide the test code and related models, and the full code will be released later.

##  1. Requirements

   - Hardware: 4-8 GPUs (better with >=11G GPU memory)
   - Software: PyTorch>=1.0.0, Python3, [tensorboardX](https://github.com/lanpa/tensorboardX),  and so on.
   - to install tensorboardX, you could do as follows:

     ```
     pip install tensorboardX
     pip install tensorboard
     ```
- install python packages: `pip install -r requirements.txt`

## 2. Dataset

### Composition-1K

Please contact Brian Price ([bprice@adobe.com](mailto:bprice@adobe.com)) requesting for the dataset. And following the instructions for preparation.


### Synthetic Unified Dataset

Since the dataset is processed from Composition-1K, you can generate it by:

- Acquire Composition-1K as metioned above
- Select SO and ST images from license-free websites, and composit them with SO and ST from Composition-1K to generate NSO and NST images (You need a manual selection to confirm the composited image are natural with multi-objects)

Finally, the folder structure should be:

```
DATA_ROOT
├── Combined_Dataset
│   ├── Test_set
│   │   │   ├── alpha
│   │   │   ├── fg
│   │   │   ├── bg
│   │   │   ├── trimaps
│   │   │   ├── ImageSets 
├── combined_4classes (the synthetic unified dataset)
│   │   ├── alphas
│   │   ├── images
│   │   │   ├── SO
│   │   │   ├── ST
│   │   │   ├── NSO
│   │   │   ├── NST
│   │   ├── trimaps
```

## 3. Model Zoo

| Methods   | Annotations  | Links | MSE  | SAD  | Grad | Conn | Notes |
| --------- | :----------: | :---: | :--: | :--: | :--: | :--: | ---- |
| UIM (box) | Bounding box | [gdrive](https://drive.google.com/file/d/14ofHr1_Z5sxSVyE9efReZtH6_HtX_CQb/view?usp=sharing) | 0.012 | 38.15 | 17.90 | 33.76 | trimap-based |
| UIM (box) | Bounding box | [gdrive](https://drive.google.com/file/d/1LvZJXQIDBj0I76C99cMy6amPn13DtkHB/view?usp=sharing) | 0.006 | 49.85  | 25.24  | 43.60  | trimap-free  |
| UIM (dextr) | Extreme points | [gdrive](https://drive.google.com/file/d/1p4FBelvRpUZlLFYkaP-49xYesdI_JDBj/view?usp=sharing) | 0.015 | 77.25 | 33.14 | 59.93 | trimap-free |
| UIM (in_point) | Foreground point | [gdrive](https://drive.google.com/file/d/1FpuHaTBDp-_9R2sO908T7Iltmtah5Snk/view?usp=sharing) | 0.077 | 265.87 | 103.81 | 142.87 | trimap-free |
| UIM (iog) | FG/BG points | [gdrive](https://drive.google.com/file/d/1bUbuQOEDtgzEEN6yKkyYfI7aUzRwPSRP/view?usp=sharing) | 0.042 | 165.92 | 74.85 | 92.18 | trimap-free |
| UIM (scribble) | Scribble | [gdrive](https://drive.google.com/file/d/1D15KR7OgZKDwVE1OE9RITLUZLC_17Zvs/view?usp=sharing) | 0.039 | 139.03 | 39.55 | 58.31 | trimap-free |

*The metrics are all tested on Composition-1K

## 4. Inference

Download pretrained models and put them under `./pretrained`.

Run on one GPU to evaluate the model, the examples are as follow:

- Test on Composition-1K:

```
cd UIM/
sh seg_matting_tool/test.sh comp1k uim_bbox
```

- Test on the synthetic unified dataset:

```
cd UIM/
sh seg_matting_tool/test.sh 4classes uim_bbox
```

## 5. License

This project is under the MIT license.

