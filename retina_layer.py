#use in cnn or 4D tensors
class retina_module(nn.Module):
    def __init__(self,inp,features,size=8,device="cpu",prob=0.7):
        super(retina_module,self).__init__()
        
        self.filter_maxprob=prob
        
        self.input_conv1= nn.Conv2d(inp,features,3,padding="same")
        self.input_conv2= nn.Conv2d(features,features,3,padding="same")
        self.out_conv= nn.Conv2d(features,features,3,padding="same")
        self.size =size
        self.masks = features
        
        self.q1_predictor= nn.Conv2d(features,1,3,padding="same")
        self.v1_predictor= nn.Conv2d(features,1,3,padding="same")
        self.act1 = nn.LeakyReLU()
        self.act2 =nn.Sigmoid()
        self.adv_pool = nn.AvgPool2d((2,2))
        self.device=device

    def forward(self,x):
        #you can add more layers
        x=self.input_conv1(x)
        x=self.act1 (x)
        x=self.input_conv2(x)
        x=self.act1 (x)
       
        #PASS SQUEEZE TENSOR
        batch_size = x.size(0)
        x =  x.view(-1, x.size(1),self.size ,self.size )  
       
        #now create new batch 
        #att_1
        at=  self.q1_predictor(x)
        att= self.v1_predictor(x)
        x1 = self.act2 (at*att).squeeze() 
        
        zeros = torch.zeros((batch_size, self.masks,self.size,self.size),device=self.device)
        mask = torch.where(x1>x1.max()*self.filter_maxprob)[0]
        if mask.size(0)>batch_size:
            sz =100
        else:
            sz = mask.size(0)
        zeros[:sz] = x[mask[sz]]
        x = zeros
        x=self.out_conv(x)
        #(batch, chanels/features, x, y)
        return x
