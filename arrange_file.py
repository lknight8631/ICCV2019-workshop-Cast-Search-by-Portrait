import json
import shutil
import os
from PIL import Image
from argparse import ArgumentParser

def make_dir(_path, _is_del = True):
    if _path[-1] != '/':
        _path = _path + '/'
    if os.path.isdir(_path):
        if _is_del == True:
            shutil.rmtree(_path)
            os.mkdir(_path)
    else:
        os.mkdir(_path)
    return _path

def cast_writer(_src_root, _root_path, _path, _obj):
    _ = make_dir(_path + 'cast', True)
    result_dict = {}
    for item in _obj:
        src = _src_root + item['img']
        dst = _ + item['id'] + '.jpg'
        # copy file
        shutil.copyfile(src, dst)
        result_dict[dst] = item['label']
    with open(_path + 'cast.json', 'w') as f:
        json.dump(result_dict, f)

def candidates_writer(_src_root, _root_path, _path, _obj):
    _ = make_dir(_path + 'candidates', True)
    result_dict = {}
    for item in _obj:
        src = _src_root + item['img']
        dst = _ + item['id'] + '.jpg'
        
        # x, y, w, h
        bbox = (item['bbox'][0], item['bbox'][1], item['bbox'][0] + item['bbox'][2], item['bbox'][1] + item['bbox'][3])
        img = Image.open(src)
        crop_img = img.crop(bbox)
        crop_img.save(dst)
#         result_dict[dst] = item['label']
        
#     with open(_path + 'candidate.json', 'w') as f:
#         json.dump(result_dict, f)


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--or_dir", default='test', help="test directory")
    args = parser.parse_args()

    data_type = args.or_dir
    json_file = './' + data_type + '.json'
    whole_path = make_dir('./' + data_type + '_id_format/', True)
    print('processing ' + data_type + ' datatset')
    # load json file
    with open(json_file, 'r') as js_file:
        data = json.loads(js_file.read())

        for idx, movie_id in enumerate(data):
            movie_whole_path = make_dir(whole_path + movie_id, True)
            cast_writer('./' + data_type + '/', whole_path, movie_whole_path, data[movie_id]['cast'])
            candidates_writer('./' + data_type + '/', whole_path, movie_whole_path, data[movie_id]['candidates'])
            
            print(idx, movie_id + ' done!')
            
