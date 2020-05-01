# Make headlines funny again   

## Description   
Understanding and predicting humor is a semantically challenging task. In a quest to
make AI agents more human-like, it becomes essential that they understand the complex social and psychological trait of humor coming very naturally to us. Although some work has been done on proposing methods and datasets for the task, very little work has been done on understanding what makes something funny. A recent work aimed at the same, has recently proposed the Humicroedit (Hossain et al.) dataset, which contains edited news headlines graded for funniness, as a step to identify causes of humor. In our work, we solve for the task of regressing funniness and predicting the funnier edited headline by leveraging the recently proposed powerful LM’s and humor heuristics-based features.   

## How to run   
 
```bash
# clone project   
git clone https://github.com/lunayach/funnyAgain.git   

# move to project folder
cd funnyAgain 

 ```   
 Next, navigate to [baselines/] and run the experiments.   
 ```bash
 Example,
# module folder
cd baselines/roberta_cached_features 

# run module 
python s_bert_trainer.py    
```
## References (for Code)
* https://github.com/PyTorchLightning/pytorch-lightning
* https://huggingface.co/transformers/

