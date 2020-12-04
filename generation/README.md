# GPT-2 recipe fine-tuner
This repository uses:
* Pytorch 1.3.0
* CUDA 10.1
* NVIDIA apex
* Huggingface Transformers 2.1.0

To provide complete pipeline starting with organized and cleaned dataset, and finishing with fine-tuned gpt-2 recipe generator

## Data
Recipe1M dataset is used and enhanced with following tags:
* <TITLE_START>
* <TITLE_END>
* <INSTR_START>
* <NEXT_STEP>
* <INSTR_END>
* <INGR_START>
* <NEXT_INGR>
* <INGR_END>
* <INPUT_START>
* <NEXT_INPUT>
* <INPUT_END>

As GPT-2 has 1024 tokens of context, the larger recipes were removed, and shorter ones were grouped so that they were 1024 tokens long.

**TODO: more data, more preprocessing**

## Training
Training was performed on RTX2060 GPU. The transformer is memory hungry and 6GB of card was barely fitting, so NVIDIA apex was used for mixed precision ( mostly fp16) calculations, which decreased the size of model by half.
Steps to train, starting from `layers1.json` Recipe1M+ file
1) Run `preparation.py` on your dataset  to prepare it. You'll get `unsupervised_train.txt` and `unsupervised_test.txt` as an output.
2) Run `tokenization` on these files to prepare training matrix of shape `(n_records, 1024)`. The output will be large, single, `unsupervised.h5` file.
3) use `train.sh` to train model. If training on Colab GPU, make sure to provide your S3 credentials in `credentials.sh` file.

## Inference
To test results of this transformer, please install `python3` with `pytorch` and `transformers` packages. Then run `play.sh`. The latest model should be downloaded from my S3, and after a while the console appear, asking you for input. Provide comma separated list of ingredients. If you don't want model to generate any more input ingredients on its own, end the input string with semicolon. Press enter and observe the transformer generating the rest of recipe for you!
