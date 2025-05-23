# AnimeRealDetect
This is test and finetune project to predict anime or real.

## Models download
Download model from huggingface [caformer_s36_v1.3_fixed](https://huggingface.co/deepghs/anime_real_cls/tree/main/caformer_s36_v1.3_fixed), and put it under folder ```model```.

## Finetune
```
python finetune.py
```

## Test
### simple inference
```
python inference.py
```
### file batch inference
```
python inference_batch.py
```
### file batch gradio inference
```
python inference_batch_gradio.py --server_port 7881
```