# StyleSRN

This repository provides the testing code for the paper "StyleSRN: Scene Text Image Super-Resolution with Text Style Embedding
" (ICCV2025). 

## Environment

![python](https://img.shields.io/badge/Python-v3.9-green.svg?style=plastic)  ![pytorch](https://img.shields.io/badge/Pytorch-v2.01-green.svg?style=plastic)  ![cuda](https://img.shields.io/badge/Cuda-v11.7-green.svg?style=plastic) 

Other possible Python packages are also needed, please refer to *requirements.txt* for more information.


##  Datasets and Pre-trained Model
- Download the STISR dataset TextZoom for real-world degradation：https://github.com/JasonBoy1/TextZoom.

- Download the Scene Text Recognition datasets, and the pre-trained weights of StyleSRN for using [PARSeq](https://github.com/baudm/parseq) as a text prior generator via this [link](https://pan.baidu.com/s/1ZMEmGcJMkaWPeWgH2EmvqA?pwd=umrg).

- Remember to modify the `val_data_dir`, `resume` and `full_ckpt` in the  `./config/super_resolution.yaml` file.



## Recognizers

In the main text, we use [CRNN](https://github.com/meijieru/crnn.pytorch), [MORAN](https://github.com/Canjie-Luo/MORAN_v2  ) and [ASTER](https://github.com/ayumiymk/aster.pytorch) to evaluate the accuracy improvement of the STISR method on downstream text recognition tasks. You can select different recognizers for evaluation by modifying the test command (e.g., `--test_model='ASTER'`). 
You can also find pre-trained weights in their repository and don’t forget to modify the corresponding recognizer path in the `./config/super_resolution.yaml` file.




## Inference
### Test on TextZoom
```
python test.py --mask --tpg PARSeq
```
### Test on manually degraded STR datasets for generalization
```
python test.py --mask --tpg PARSeq --general
```

## Citation

If you find our work helpful, please consider citing us.

```
@InProceedings{Yuan_2025_ICCV,
    author    = {Yuan, Shengrong and Wang, Runmin and Hao, Ke and Ma, Xuqi and Gao, Changxin and Liu, Li and Sang, Nong},
    title     = {StyleSRN: Scene Text Image Super-Resolution with Text Style Embedding},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2025},
    pages     = {18693-18702}
}
```



## Acknowledgement

The code of this work is based on [TBSRN](https://github.com/FudanVI/FudanOCR/tree/main/scene-text-telescope), [TATT](https://github.com/mjq11302010044/TATT), [C3-STISR](https://github.com/JingyeChen/C3-STISR) and [LEMMA](https://github.com/csguoh/LEMMA). Thanks for your contributions.
