#use in cnn or 4D tensors

class retina_module(nn.Module):
    def __init__(self,inp,features,size= 8,device="cpu"):
        super(retina_module,self).__init__()
        
        self.out_conv= nn.Conv2d(inp,features,3,padding="same")
        self.size =(int((size)/2))
        self.masks = features
        self.x_predictor= nn.Conv2d(features,features,3,padding="same")
        self.y_predictor= nn.Conv2d(features,features,3,padding="same")
        #optional
        #self.hiden_conv =  nn.Conv2d(features,features,2,padding="same")
        self.act1 = nn.LeakyReLU()
        self.act2 =nn.LeakyReLU()
        self.device=device
        #class out 
        
    
        
        
    def forward(self,x
        #you can add more layers
        x=self.out_conv(x)
        x= self.act2(x)
        x_p=self.x_predictor(x)
        y_p=self.y_predictor(x)
        #compute retina location
        mask = (x_p*y_p)+x
        mask = self.act1(mask)
        retina = []
        index = 0
        padded = torch.zeros(mask.size(0),self.masks,self.size*2,self.size*2,device=self.device).float()
        #iterate over batch
        for batch in  mask.split(1,dim=0):
            msk_ind = 0
            #iterate over feature maps for the retina
            for mask_F in mask.split(1,dim=1):
              #  print(mask_F.shape)
                _,_,x_I,y_Y = torch.where(mask_F==mask_F.max())
                if x_I.shape[0]>1:
                    x_I = x_I[0]
                    y_Y = y_Y[0]

                feature_map =  x[index][msk_ind][int(x_I-self.size) : int(x_I+self.size) , int(y_Y-self.size) : int(y_Y+self.size)]

                sizze =feature_map.shape
                padded[index,msk_ind,:sizze[0],:sizze[1]] = feature_map 

                msk_ind = msk_ind+1
            index = index+1
        retina.append(padded)

        mask = torch.cat(retina,axis=0)
        #(batch, chanels/features, x, y)
        return mask
                
    
