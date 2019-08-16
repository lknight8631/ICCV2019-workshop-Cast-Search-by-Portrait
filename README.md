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
  * Face model affinity matrix: https://blog.csdn.net/u012067966/article/details/50736647
  * IDE model affinity matrix:  https://blog.csdn.net/u012067966/article/details/50736647
* Please put test.json, test folder in the same folder with the arrange_file.py.

```
python3 arrange_file.py --or_dir='test'
python3 ensemble_ppcc.py --test_dir='./test_id_format' --output_name='the path you want'
```   

## Result

mAP: 0.8137

## Achievement

We get the 3rd place in WIDER Face & Person Challenge 2019 - Track 3: Cast Search by Portrait.

## more information
If you have any problems of our work, you may
* Contact Bing-Jhang Lin by e-mail (lknight8631@gmail.com)
