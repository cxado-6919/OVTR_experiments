import torch
from PIL import Image
import numpy as np

from lvis import LVIS
import random
import operator

import os
import clip

json_file = './data/lvis_v1/annotations/lvis_v1_train.json'
img_root = './data/lvis_v1'
# save_region = './regions'
region_ratio = 1.2
image_embedding_save_path = './model_zoo/'
test_num = 1203
# sample_num = [20, 50, 100, 200]
sample_num = [50, 200, 500, 1000]

class Make_image_embedding():
    def __init__(self):
        anno = LVIS(json_file)
        self.cat2img = anno.cat_img_map
        self.img_map = anno.img_ann_map
        self.imgs = anno.imgs
        self.embeding_list = []
        self.ratio_m = (region_ratio - 1)/2 
        self.ratio_p = 1 + (region_ratio - 1)/2 
        self.device = "cuda"
        self.clip_model, self.preprocess = clip.load('ViT-B/32', self.device)
        self.feature_all = []
        self.preprocessed_category = []
        

    def _gen_img_ebd(self):

        cat2img = self.cat2img

        # for category in cat2img:
        for cat_id in range(len(cat2img)):
            category = cat_id+1
            self.preprocessed_category = []

            if len(cat2img[category]) <sample_num[0]:
                sample_index = list(range(len(cat2img[category])))
            elif len(cat2img[category]) < 200:
                sample_index = random.sample(range(len(cat2img[category])),sample_num[0])
                sample_index.sort()
            elif len(cat2img[category]) < 1000:
                sample_index = random.sample(range(len(cat2img[category])),sample_num[1])
                sample_index.sort()
            elif len(cat2img[category]) < 6000:
                sample_index = random.sample(range(len(cat2img[category])),sample_num[2])
                sample_index.sort()
            else:
                sample_index = random.sample(range(len(cat2img[category])),sample_num[3])
                sample_index.sort()
            self.load_img_from_map(sample_index, cat2img[category], category)
            
            # torch.cuda.empty_cache()
            if len(self.preprocessed_category) == 0:
                print(f"category {category} is NOT OK.")
                continue
            preprocessed_category = torch.stack(self.preprocessed_category).to(self.device)
            feature_category = self.clip_model.encode_image(preprocessed_category).float()
            feature_category = torch.nn.functional.normalize(feature_category, p=2, dim=1)
            mean_feature = feature_category.mean(dim=0).unsqueeze(0)
            self.feature_all.append(mean_feature)
            print(f"category {category} is OK.")
            if len(self.feature_all) == test_num:
                break
        self.gather_embedding()
                
    def load_img_from_map(self, sample_index, img_index, category_id):
        if len(sample_index) == 1 :
            img_id = [img_index[sample_index[0]]]
        else:
            img_id = list(operator.itemgetter(*sample_index)(img_index))

        instance_count = 0
        box_list = []
        region_id_list = []
        for j in range(len(img_id)):
            i = img_id[j]
            for instance_id in range(instance_count, len(self.img_map[i])):
                instance_count += 1
                if self.img_map[i][instance_id]['category_id'] == category_id:
                    bbox = self.img_map[i][instance_id]['bbox']
                    box_list.append(bbox)
                    region_id_list.append(j+1)
                    if j != (len(img_id)-1):
                        if img_id[j+1] != i:
                            self.get_image_region(np.array(box_list).astype(np.float32), i, category_id, region_id_list)
                            instance_count = 0
                            box_list = []
                            region_id_list = []
                    else:
                        self.get_image_region(np.array(box_list).astype(np.float32), i, category_id, region_id_list)
                    break 

    def get_image_region(self, boxs, image_id, category_id, region_id_list):
        img_path = img_root + self.imgs[image_id]['coco_url'].replace('http://images.cocodataset.org', '')
        img = Image.open(img_path)
        image_np = np.array(img)
        img = Image.fromarray(image_np)
        img_shape = img.size

        b_boxs = np.zeros_like(boxs)
        b_boxs[:,0] = boxs[:,0] - boxs[:,2] * self.ratio_m
        b_boxs[:,2] = boxs[:,0] + boxs[:,2] * self.ratio_p
        b_boxs[:,1] = boxs[:,1] - boxs[:,3] * self.ratio_m
        b_boxs[:,3] = boxs[:,1] + boxs[:,3] * self.ratio_p
        b_boxs = np.around(b_boxs)
        box_x = np.clip(b_boxs[:,0:4:2],0,img_shape[0])
        box_y = np.clip(b_boxs[:,1:4:2],0,img_shape[1])
        boxs[:,0] = box_x[:,0]
        boxs[:,1] = box_y[:,0]
        boxs[:,2] = box_x[:,1]
        boxs[:,3] = box_y[:,1]

        for i, box in enumerate(boxs):
            if (box[0] > (box[2] - 2)) or (box[1] > (box[3] - 2)):
                continue
            cropped = img.crop(box)  # (left, upper, right, lower)
            # os.makedirs(os.path.join(save_region,f'{category_id}c'),exist_ok=True)
            # cropped.save(os.path.join(save_region,f'{category_id}c',f'{region_id_list[i]}_{image_id}img_region.jpg'))

            #cropped
            cropped_n = self.preprocess(cropped)
            self.preprocessed_category.append(cropped_n)

    def gather_embedding(self):
        clip_image_embedding = torch.cat(self.feature_all, dim=0)
        torch.save(clip_image_embedding,os.path.join(image_embedding_save_path,'clip_image_embeddings.pt'))


if __name__ == "__main__":
    with torch.no_grad():
        a = Make_image_embedding()
        a._gen_img_ebd()
