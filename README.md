# Finetuning Folktexts 

This is a repository for finetuning folktexts. 

## Setup 

```bash
pip install -r requirements.txt
```

## Preparing the data 

Since torchtexts library is not meant for finetuning, we need to modify it a little bit. 

```bash
python src/prompts.py
```

This will create a `train_data` directory with the data in it. 


## Finetuning a model

We use the `torchtune` library to finetune a model. 

In `configs/`, we have the different configs for finetuning the model. 

The most relevant keys are: 

- `output_dir`: the directory to save the finetuned model. 
- `checkpointer.checkpoint_dir`: the directory to the pretrained model weights. 
- `dataset.source`: the directory to the data. 
- `train_on_input`: whether to train on the input or the output. 
- `conversation_style`: the style of the conversation check here for information [here](https://pytorch.org/torchtune/0.3/basics/chat_datasets.html).

There are a few important things that can affect finetuning:

- do we finetune on all data yay or nay. 
- do we do full finetuning or lora
- is the data chat or not. 

## Evaluating the model

```bash
python src/test_model.py
```

This will evaluate the model on the test set. 

NOTE: the model should actually probably evaluated using the underlying folktexts library.

Checkout `basic setup` [here](https://github.com/socialfoundations/folktexts/tree/main)




# OLD



# TODO

- [x] integrate prompt construction from folktexts 
- [x] connect data pipeline with torch tune 
- [x] set up torch tune recipe for finetuning 
- [ ] finetune model in three contexts
     - [x] basic task full updates (10,000 data points)
     - [x] basic task only predict final value (next week)
          - this requires further data integration 
     - [ ] hash the actual values. (next week?)
- [x] connect the evaluations pipeline from folktexts 
- [ ] build viz pipeline 
- [ ] connect the ECE calibration tools from folktexts.

codebase:
- data creation. integrate with folktexts. split into train/test/val. 
- finetuning configs. 
   - it non it settings. 
   - random weights setting 
- extracting top logits for all sampled tokens. 
- test set 

4 Feb 2024
- [x] add in the instruct model. compare it with non-it model
- [x] use a smaller model 
- [ ] calibration ECE comparisons across settings
- [ ] weird properties in the unbalanced setting. model learns to always predict A. maybe bc its smaller model?
- [ ] full prompt vs final prompt comparison. tried this, it's a bit weird. 
- [ ] prepare the data for interpolation of results. 
- [x] reinitialize model weights 

observations:
- finetuning isn't enough. not a lot of stability in terms of model generations. 

- the more the weight gets put on a smaller 
- ablate on the qlora and do full lora.
- bit counting. continuous integration. 


Proposed studies:
1. Finetuning vs. inference gap (few-shot comparison from paper).
2. Recapitulate Alessandra's findings with hash values. 
3. Reinitialize model weights. 
4. Scaling data / scaling of tuneable parameters. 