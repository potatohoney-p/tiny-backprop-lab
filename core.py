import math
import random
def sigmoid(x):
    if x>=0: return 1/(1+math.exp(-x))
    e=math.exp(x); return e/(1+e)
class Network:
    def __init__(self,inputs=2,hidden=4,seed=5):
        if inputs<1 or hidden<1: raise ValueError('Positive layer sizes required')
        self.inputs=inputs; self.hidden=hidden; rng=random.Random(seed)
        self.weights=[rng.uniform(-1,1) for _ in range(hidden*(inputs+1)+hidden+1)]
    def forward(self,row):
        if len(row)!=self.inputs or any(not math.isfinite(x) for x in row): raise ValueError('Invalid input row')
        h=[sigmoid(sum(self.weights[j*(self.inputs+1)+i]*x for i,x in enumerate(row))+self.weights[j*(self.inputs+1)+self.inputs]) for j in range(self.hidden)]
        offset=self.hidden*(self.inputs+1)
        return h,sigmoid(sum(self.weights[offset+j]*v for j,v in enumerate(h))+self.weights[-1])
    def gradients(self,row,target):
        if target not in (0,1): raise ValueError('Binary target required')
        h,y=self.forward(row); delta=(y-target)*y*(1-y); offset=self.hidden*(self.inputs+1)
        result=[]
        for j,v in enumerate(h):
            d=delta*self.weights[offset+j]*v*(1-v)
            result.extend([d*x for x in row]+[d])
        return result+[delta*v for v in h]+[delta]
    def predict(self,row): return self.forward(row)[1]
    def loss(self,row,target): return 0.5*(self.predict(row)-target)**2
    def train(self,rows,targets,epochs=3000,rate=1.0):
        if not rows or len(rows)!=len(targets) or epochs<0 or not math.isfinite(rate) or rate<=0: raise ValueError('Invalid training parameters')
        for row,target in zip(rows,targets): self.gradients(row,target)
        for _ in range(epochs):
            for row,target in zip(rows,targets):
                self.weights=[w-rate*g for w,g in zip(self.weights,self.gradients(row,target))]
