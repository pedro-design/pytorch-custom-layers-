#use in cnn or 4D tensors
class retina_module(nn.Module):
    def __init__(self,inp,features,inp_sz,size=8,lstm_features=32 ,device="cpu",prob=0.85):
        super(retina_module,self).__init__()
        
        self.filter_maxprob=prob
        self.size =size
        self.stride= int((size-1)/2)
        self.masks = features
        self.lstm_F = lstm_features
        self.depth_conv = nn.Conv2d(inp, features, kernel_size=3, groups=inp,padding="same")
        self.act1 = nn.LeakyReLU()
        self.act2 =nn.LeakyReLU()
        self.lstm_0 =  nn.LSTMCell(inp_sz, lstm_features, bias=True)
        self.lstm_1 =  nn.LSTMCell(lstm_features, 2, bias=True)

    def forward(self,x):
        #you can add more layers
        B, C, H, W = x.shape
        start = []
        padded = F.pad(x, (self.size // 2, self.size // 2, self.size // 2, self.size // 2))
        index = 0
        for image in x.split(1,dim=0):
            state_H1 = torch.zeros((image.size(2), self.lstm_F),device=device)
            state_C1 = torch.zeros((image.size(2), self.lstm_F),device=device)
            
            state_H2 = torch.zeros((image.size(2), 2),device=device)
            state_C2 = torch.zeros((image.size(2) ,2),device=device)
            features_retina = []
            step_IND = 0
            for step in image.split(1,dim=1):
                step = step.squeeze()
                state_H1,state_C1= self.lstm_0(step,(state_H1,state_C1 ))
                state_H2,state_C2= self.lstm_1(state_H1,(state_H2,state_C2 ))
                out_ =state_H2.argmax(axis=0)
                features_retina.append(padded[index,step_IND, out_[1] : out_[1]+self.size, out_[ 0] : out_[ 0]+self.size].unsqueeze(0).unsqueeze(0))
                step_IND = step_IND+1
            out=  torch.cat(features_retina,dim=0)
            index = index+1
         
            start.append(out)

        x = torch.cat(start,axis=0)
        x=  self.depth_conv (x)
        return x
