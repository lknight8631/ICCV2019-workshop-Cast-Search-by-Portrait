# ICCV2019_workshop_Cast-Search-by-Portrait

## How to Reproduce the result

* Download our affinity matrix first.
  * Face model affinity matrix: https://blog.csdn.net/u012067966/article/details/50736647
  * IDE model affinity matrix:  https://blog.csdn.net/u012067966/article/details/50736647
* Please put test.json, test folder in the same folder with the arrange_file.py.

```
python3 arrange_file.py --or_dir='test'
python3 ensemble_ppcc.py --test_dir='./test_id_format' --output_name='the path you want'
```   


## Requirement
  pytorch=1.2
