

#GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. 
#This means it was pretrained on the raw texts only, with no humans labelling them in any way 
#(which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. 
#More precisely, it was trained to guess the next word in sentences.
#More precisely, inputs are sequences of continuous text of a certain length and the targets are the same sequence, shifted one token (word or piece of word) to the right. 
#The model uses internally a mask-mechanism to make sure the predictions for the token i only uses the inputs from 1 to i but not the future tokens.
#This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks. 
#The model is best at what it was pretrained for however, which is generating texts from a prompt.!pip install transformers

import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id) #to stay at the same length , Will then be ignored by attention mechanisms or loss computation

sentence = 'YouTube Title: AI learns to' #prompet

input_ids = tokenizer.encode(sentence, return_tensors='pt')

input_ids

# generate text until the output length (which includes the context length) reaches 50

output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
# no_repeat_ngram_size = The size of an n-gram that cannot occur more than once. ( 0=infinity) bad_words
#  early_stopping =to stop training at the point when performance on a validation dataset starts to degrade
output
# num_beams = beam search
print(tokenizer.decode(output[0], skip_special_tokens=True))
# skip_special_tokens = (`bool`, *optional*, defaults to `False`):. Whether or not to remove special tokens in the decoding
