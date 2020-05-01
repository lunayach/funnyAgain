## MNIST Baseline    
In this readme, give instructions on how to run your code.   

    def __init__(self, hparams):
        super(S_BERT_Regression, self).__init__()
        # not the best model...
        # self.hparams = hparams
        # self.l1 = torch.nn.Linear(768*2, 256)
        # self.l2 = torch.nn.Linear(256, 128)
        # self.l3 = torch.nn.Linear(128, 1)
        # self.dropout = torch.nn.Dropout(p=0.2)

        self.hparams = hparams
        self.l1 = torch.nn.Linear(768 * 3, 768)
        self.l2 = torch.nn.Linear(768, 256)
        self.l3 = torch.nn.Linear(256, 1)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        f1 = self.dropout(torch.relu(self.l1(x.view(x.size(0), -1))))
        f2 = torch.relu(self.l2(f1))
        out = self.l3(f2)

        # f1 = torch.relu(self.l1(x.view(x.size(0), -1)))
        # out = self.l2(f1)

        return out

#### CPU   
```bash   
python mnist_baseline_trainer.py     
```

#### Multiple-GPUs   
```bash   
python mnist_baseline_trainer.py --gpus '0,1,2,3'  
```   

#### On multiple nodes   
```bash  
python mnist_baseline_trainer.py --gpus '0,1,2,3' --nodes 4  
```   
