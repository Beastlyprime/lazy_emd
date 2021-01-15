# Lazy EMD
This repository contains a primary implementation of the Lazy Earth Mover's Distance (Lazy EMD).

## Demo
Our demo is on http://ijcai.ruibinxiong.cn, please use Chrome (others, e.g. Safari may encounter problems.) It may take a long time (> 15s) to open the website and submit the first example, thanks for your patience! 

## Requirements
- python (==3.x)
- bert_score (>=0.2.2)
- POT (>=0.6.0)

## Usage
The simplest way is to copy `score.py` and `utils.py` to your `bert_score` file, `unbalanced.py` to your `ot` file.
The usage is then similar to BERTScore. See `example.py` for an example.

If you find this repo useful, please cite:
```
@inproceedings{lazy-emd,
  title     = {Evaluating Natural Language Generation via Unbalanced Optimal Transport},
  author    = {Chen, Yimeng and Lan, Yanyan and Xiong, Ruibin and Pang, Liang and Ma, Zhiming and Cheng, Xueqi},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere}	
  pages     = {3730--3736},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/516},
  url       = {https://doi.org/10.24963/ijcai.2020/516},
}
```

## Acknowledgement
This repo benefits a lot from [BERTScore](https://github.com/Tiiiger/bert_score) and [POT](https://github.com/PythonOT/POT).
