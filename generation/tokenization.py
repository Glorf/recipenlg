from transformers import GPT2Tokenizer
import h5py
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=False)
special_tokens = {
    "additional_special_tokens": [
        "<TITLE_START>",
        "<TITLE_END>",
        "<INSTR_START>",
        "<NEXT_INSTR>",
        "<INSTR_END>",
        "<INGR_START>",
        "<NEXT_INGR>",
        "<INGR_END>",
        "<RECIPE_START>",
        "<RECIPE_END>",
        "<INPUT_START>",
        "<INPUT_END>",
        "<NEXT_INPUT>"
    ]
}

tokenizer.add_special_tokens(special_tokens)

end_token_id = tokenizer.convert_tokens_to_ids(["<RECIPE_END>"])[0]

hf = h5py.File("unsupervised.h5", "w")
for filename in ["test", "train"]:
    out_np = []
    data = open("unsupervised_"+filename+".txt", "r")
    num = 0
    rows = 0
    last=[]
    for line in data:
        num+=1
        if num%10000 == 0:
            print("Read "+str(num)+" Written: "+str(rows))

        text_tokens = tokenizer.tokenize(line)
        if len(text_tokens) > 1024: #Recipe won't fit the model
            continue

        text_tokens_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        if (len(last) + len(text_tokens_ids)) <= 1024:
            last+=text_tokens_ids
        else:
            while len(last) < 1024:
                last.append(end_token_id)
            out_np.append(last)
            last=text_tokens_ids
            rows+=1
    out_mat = np.matrix(out_np)
    print(out_mat.shape)
    hf.create_dataset(filename, data=out_mat)
hf.close()
