#use in cnn or 4D tensors
class retina_module(nn.Module):
    def __init__(self,inp,features,inp_sz,size=8,lstm_features=32 ,device="cpu"):
        super(retina_module,self).__init__()
        
        
        self.size =size
        self.stride= int((size-1)/2)
        self.masks = features
        self.lstm_F = lstm_features
        
        self.depth_conv = nn.Conv2d(inp, features, kernel_size=3, groups=inp,padding="same")
        self.act1 = nn.LeakyReLU()
        self.act2 =nn.LeakyReLU()
        self.lstm_0 =  nn.Transformer(d_model=inp_sz, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=lstm_features)
        self.position_predictor = nn.Linear(inp_sz,2)
        self.set = 0
        self.device=device

    def forward(self,x):

        B, C, H, W = x.shape

        start = []
        padded = nn.functional.pad(x, (self.size // 2, self.size // 2, self.size // 2, self.size // 2))
        index = 0
        for image in x.split(1,dim=0):
            features_retina = []
            step_IND = 0
            image = image[0]
            state_C2 =  self.lstm_0(image,image )
            state_C2 = self.position_predictor(state_C2)
            for coords in state_C2.split(1,dim=0):
                coords = coords[0].argmax(axis=0)
                a=padded[index,step_IND, coords[1] : coords[1]+self.size, coords[ 0] : coords[ 0]+self.size]
                
                features_retina.append(padded[index,step_IND, coords[0] : coords[0]+self.size, coords[ 1] : coords[ 1]+self.size].unsqueeze(0).unsqueeze(0))
                step_IND = step_IND+1
            out=  torch.cat(features_retina,dim=1)
            index = index+1
            start.append(out)
        x = torch.cat(start,axis=0)
        x=  self.depth_conv (x)
        return x
