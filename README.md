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