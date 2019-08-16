# ICCV2019_workshop_Cast-Search-by-Portrait

## Requirement
> python=3.6  
> pytorch=1.2  
> torchvision  
> numpy=1.16.2  
> Pillow  
> The Python Standard Library  
  
## How to Reproduce the result

* Download our affinity matrix first.
  * Face model affinity matrix: https://drive.google.com/file/d/1IF1lZmdtCcj7Y5E-jicDh6llb4S5Ph7K/view?usp=sharing  
  * IDE model affinity matrix:  https://drive.google.com/file/d/1LzD9a2rcPh3Io6uMIvU4_65_YK3zHaCC/view?usp=sharing  
* Please put test.json, test data in the same folder with the arrange_file.py.

```
python3 arrange_file.py --or_dir='test'
python3 ensemble_ppcc.py --test_dir='./test_id_format' --output_name='the path you want'
```   

## Result

mAP: 0.8137

## Achievement

We get the 3rd place in WIDER Face & Person Challenge 2019 - Track 3: Cast Search by Portrait.

## References
```
@InProceedings{Cao18,
author = "Cao, Q. and Shen, L. and Xie, W. and Parkhi, O. M. and Zisserman, A.",
title  = "VGGFace2: A dataset for recognising faces across pose and age",
booktitle = "International Conference on Automatic Face and Gesture Recognition",
year  = "2018"}

@article{zhang2016joint,
  title={Joint face detection and alignment using multitask cascaded convolutional networks},
  author={Zhang, Kaipeng and Zhang, Zhanpeng and Li, Zhifeng and Qiao, Yu},
  journal={IEEE Signal Processing Letters},
  volume={23},
  number={10},
  pages={1499--1503},
  year={2016},
  publisher={IEEE}
}

@inproceedings{bulat2017far,
  title={How far are we from solving the 2d \& 3d face alignment problem?(and a dataset of 230,000 3d facial landmarks)},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1021--1030},
  year={2017}
}

@inproceedings{huang2018person,
    title={Person Search in Videos with One Portrait Through Visual and Temporal Links},
    author={Huang, Qingqiu and Liu, Wentao and Lin, Dahua},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    pages={425--441},
    year={2018}
}

@inproceedings{luo2019bag,
  title={Bag of tricks and a strong baseline for deep person re-identification},
  author={Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={0--0},
  year={2019}
}

@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}

@inproceedings{hu2018squeeze,
  title={Squeeze-and-excitation networks},
  author={Hu, Jie and Shen, Li and Sun, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7132--7141},
  year={2018}
}

@article{zhong2017re,
  title={Re-ranking Person Re-identification with k-reciprocal Encoding},
  author={Zhong, Zhun and Zheng, Liang and Cao, Donglin and Li, Shaozi},
  booktitle={CVPR},
  year={2017}
}
```

## more information
If you have any problems of our work, you may
* Contact Bing-Jhang Lin by e-mail (lknight8631@gmail.com)
