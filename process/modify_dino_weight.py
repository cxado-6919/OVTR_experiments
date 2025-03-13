import torch

dino_weight = './model_zoo/checkpoint0033_4scale.pth'
checkpoint = torch.load(dino_weight)

ori_check=checkpoint['model']
copy_list=['transformer.decoder.layers.0.linear1.weight', 'transformer.decoder.layers.0.linear1.bias', 
           'transformer.decoder.layers.0.linear2.weight', 'transformer.decoder.layers.0.linear2.bias', 
           'transformer.decoder.layers.0.norm3.weight', 'transformer.decoder.layers.0.norm3.bias', 
           'transformer.decoder.layers.1.linear1.weight', 'transformer.decoder.layers.1.linear1.bias', 
           'transformer.decoder.layers.1.linear2.weight', 'transformer.decoder.layers.1.linear2.bias', 
           'transformer.decoder.layers.1.norm3.weight', 'transformer.decoder.layers.1.norm3.bias', 
           'transformer.decoder.layers.2.linear1.weight', 'transformer.decoder.layers.2.linear1.bias', 
           'transformer.decoder.layers.2.linear2.weight', 'transformer.decoder.layers.2.linear2.bias', 
           'transformer.decoder.layers.2.norm3.weight', 'transformer.decoder.layers.2.norm3.bias', 
           'transformer.decoder.layers.3.linear1.weight', 'transformer.decoder.layers.3.linear1.bias', 
           'transformer.decoder.layers.3.linear2.weight', 'transformer.decoder.layers.3.linear2.bias', 
           'transformer.decoder.layers.3.norm3.weight', 'transformer.decoder.layers.3.norm3.bias', 
           'transformer.decoder.layers.4.linear1.weight', 'transformer.decoder.layers.4.linear1.bias', 
           'transformer.decoder.layers.4.linear2.weight', 'transformer.decoder.layers.4.linear2.bias', 
           'transformer.decoder.layers.4.norm3.weight', 'transformer.decoder.layers.4.norm3.bias', 
           'transformer.decoder.layers.5.linear1.weight', 'transformer.decoder.layers.5.linear1.bias', 
           'transformer.decoder.layers.5.linear2.weight', 'transformer.decoder.layers.5.linear2.bias', 
           'transformer.decoder.layers.5.norm3.weight', 'transformer.decoder.layers.5.norm3.bias', ]
            # 'transformer.decoder.norm.weight', 'transformer.decoder.norm.bias'
        
paste_list=['transformer.decoder.layers.0.linear3.weight', 'transformer.decoder.layers.0.linear3.bias', 
            'transformer.decoder.layers.0.linear4.weight', 'transformer.decoder.layers.0.linear4.bias', 
            'transformer.decoder.layers.0.norm5.weight', 'transformer.decoder.layers.0.norm5.bias', 
            'transformer.decoder.layers.1.linear3.weight', 'transformer.decoder.layers.1.linear3.bias', 
            'transformer.decoder.layers.1.linear4.weight', 'transformer.decoder.layers.1.linear4.bias', 
            'transformer.decoder.layers.1.norm5.weight', 'transformer.decoder.layers.1.norm5.bias', 
            'transformer.decoder.layers.2.linear3.weight', 'transformer.decoder.layers.2.linear3.bias', 
            'transformer.decoder.layers.2.linear4.weight', 'transformer.decoder.layers.2.linear4.bias', 
            'transformer.decoder.layers.2.norm5.weight', 'transformer.decoder.layers.2.norm5.bias', 
            'transformer.decoder.layers.3.linear3.weight', 'transformer.decoder.layers.3.linear3.bias', 
            'transformer.decoder.layers.3.linear4.weight', 'transformer.decoder.layers.3.linear4.bias', 
            'transformer.decoder.layers.3.norm5.weight', 'transformer.decoder.layers.3.norm5.bias', 
            'transformer.decoder.layers.4.linear3.weight', 'transformer.decoder.layers.4.linear3.bias', 
            'transformer.decoder.layers.4.linear4.weight', 'transformer.decoder.layers.4.linear4.bias', 
            'transformer.decoder.layers.4.norm5.weight', 'transformer.decoder.layers.4.norm5.bias', 
            'transformer.decoder.layers.5.linear3.weight', 'transformer.decoder.layers.5.linear3.bias', 
            'transformer.decoder.layers.5.linear4.weight', 'transformer.decoder.layers.5.linear4.bias', 
            'transformer.decoder.layers.5.norm5.weight', 'transformer.decoder.layers.5.norm5.bias', ]
            # 'transformer.decoder.norm_inter.weight', 'transformer.decoder.norm_inter.bias'


copy_meta=['transformer.decoder.layers.0.linear1', 
           'transformer.decoder.layers.0.linear2', 
           'transformer.decoder.layers.0.norm3', 
           'transformer.decoder.layers.1.linear1', 
           'transformer.decoder.layers.1.linear2', 
           'transformer.decoder.layers.1.norm3', 
           'transformer.decoder.layers.2.linear1', 
           'transformer.decoder.layers.2.linear2', 
           'transformer.decoder.layers.2.norm3', 
           'transformer.decoder.layers.3.linear1', 
           'transformer.decoder.layers.3.linear2', 
           'transformer.decoder.layers.3.norm3', 
           'transformer.decoder.layers.4.linear1', 
           'transformer.decoder.layers.4.linear2', 
           'transformer.decoder.layers.4.norm3', 
           'transformer.decoder.layers.5.linear1', 
           'transformer.decoder.layers.5.linear2', 
           'transformer.decoder.layers.5.norm3', ]
  
paste_meta=['transformer.decoder.layers.0.linear3', 
            'transformer.decoder.layers.0.linear4', 
            'transformer.decoder.layers.0.norm5', 
            'transformer.decoder.layers.1.linear3', 
            'transformer.decoder.layers.1.linear4', 
            'transformer.decoder.layers.1.norm5', 
            'transformer.decoder.layers.2.linear3', 
            'transformer.decoder.layers.2.linear4', 
            'transformer.decoder.layers.2.norm5', 
            'transformer.decoder.layers.3.linear3', 
            'transformer.decoder.layers.3.linear4', 
            'transformer.decoder.layers.3.norm5', 
            'transformer.decoder.layers.4.linear3', 
            'transformer.decoder.layers.4.linear4', 
            'transformer.decoder.layers.4.norm5', 
            'transformer.decoder.layers.5.linear3', 
            'transformer.decoder.layers.5.linear4', 
            'transformer.decoder.layers.5.norm5', ]

for k_copy, k_paste in zip(copy_list, paste_list):  
    assert k_copy in ori_check, f"Key {k_copy} does not exist in ori_check."  
    ori_check[k_paste] = ori_check[k_copy]  

for k_copy, k_paste in zip(copy_meta, paste_meta):  
    assert k_copy in ori_check._metadata, f"Key {k_copy} does not exist in ori_check."  
    ori_check._metadata[k_paste] = ori_check._metadata[k_copy]  

checkpoint['model']=ori_check
torch.save(checkpoint, './model_zoo/dino_ep33_4scale_double_feedforward.pth')
