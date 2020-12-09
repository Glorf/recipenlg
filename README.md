# RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation

This is an archive of code which was used to produce dataset and results available in our INLG 2020 paper: [RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation](https://www.aclweb.org/anthology/2020.inlg-1.4.pdf)

## What's exciting about it?

The dataset we publish contains 2231142 cooking recipes (>2 millions). It's processed in more careful way and provides more samples than any other dataset in the area.

## Where is the dataset?

Please visit the website of our project: [recipenlg.cs.put.poznan.pl](https://recipenlg.cs.put.poznan.pl/) to download it.

## Where are your models?

The pyTorch model is available in HuggingFace model hub as [mbien/recipenlg](https://huggingface.co/mbien/recipenlg). You can therefore easily import it into your solution as follows:

```
from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("mbien/recipenlg")
model = AutoModelWithLMHead.from_pretrained("mbien/recipenlg")
```

You can also check the generation performance interactively on our website (link above).  
The SpaCy NER model is available in the `ner` directory

## Could you explain X and Y?

Yes, sure! If you feel some information is missing in our paper, please check first in our [thesis](https://www.researchgate.net/publication/345308878_Cooking_recipes_generator_utilizing_a_deep_learning-based_language_model), which is much more detailed. In case of further questions, you're invited to send us a github issue, we will respond as fast as we can!

## How to run the code?

We worked on the project interactively, and our core result is a new dataset. That's why the repo is rather a set of loosely connected python files and jupyter notebooks than a working runnable solution itself. However if you feel some part crucial for the reproduction is missing or you are dedicated to make the experience smoother, send us a feature request or (preferably), a pull request.
