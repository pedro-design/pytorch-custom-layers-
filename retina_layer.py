#use in cnn or 4D tensors
class retina_module(nn.Module):
    def __init__(self,inp,features,size= 8,device="cpu"):
        super(retina_module,self).__init__()
        
        
        self.input_conv1= nn.Conv2d(inp,features,3,padding="same")
        self.input_conv2= nn.Conv2d(features,features,3,padding="same")
        self.out_conv= nn.Conv2d(features,features,3,padding="same")
        self.size =(int((size)/2))
        self.masks = features
        self.x_predictor= nn.Conv2d(features,features,3,padding="same")
        self.y_predictor= nn.Conv2d(features,features,3,padding="same")
        self.hiden_prob_conv = nn.Conv2d(1,features,self.size,padding="same")
        
        #optional
        #self.hiden_conv =  nn.Conv2d(features,features,2,padding="same")
        self.act1 = nn.LeakyReLU()
        self.act2 =nn.LeakyReLU()
        self.device=device

    def forward(self,x):
        #you can add more layers
        x=self.input_conv1(x)
        x=self.act1 (x)
        x=self.input_conv2(x)
        x=self.act1 (x)
       
        x= self.act2(x)
        x_p=self.x_predictor(x)
        y_p=self.y_predictor(x)
        #compute retina mask
        mask = (x_p*y_p)+x
        mask = self.act1(mask)
        tam  = self.size*2 #retina size
        reshape_mask = mask.reshape(mask.size(0),-1,tam,tam).max(axis=1).values.unsqueeze(0)    
        reshape_mask =reshape_mask.permute(1,0,2,3)
        x = self.hiden_prob_conv(reshape_mask)
        x=self.out_conv(x)
        #(batch, chanels/features, x, y)
        return x
