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

## more information
If you have any problems of our work, you may
* Contact Bing-Jhang Lin by e-mail (lknight8631@gmail.com)
