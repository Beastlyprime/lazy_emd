# Lazy EMD
This repository contains a primary implementation of the Lazy Earth Mover's Distance (Lazy EMD).

## Requirements
- python (==3.x)
- bert_score (>=0.2.2)
- POT (>=0.6.0)

## Usage
The simplest way is to copy `score.py` and `utils.py` to your `bert_score` file, `unbalanced.py` to your `ot` file.
The usage is then similar to BERTScore. See `sample.py` for an example.

## Acknowledgement
This repo benefits a lot from [BERTScore](https://github.com/Tiiiger/bert_score) and [POT](https://github.com/PythonOT/POT).
