import json
import math
from core import Network
def gradient_error(network,row,target,epsilon=0.00001):
    if epsilon<=0: raise ValueError('Positive epsilon required')
    analytic=network.gradients(row,target); errors=[]
    for i,w in enumerate(network.weights):
        try:
            network.weights[i]=w+epsilon; plus=network.loss(row,target)
            network.weights[i]=w-epsilon; minus=network.loss(row,target)
        finally: network.weights[i]=w
        errors.append(abs((plus-minus)/(2*epsilon)-analytic[i]))
    return max(errors)
def dumps(network): return json.dumps({'inputs':network.inputs,'hidden':network.hidden,'weights':network.weights},sort_keys=True)
def loads(text):
    data=json.loads(text); n=Network(data['inputs'],data['hidden'])
    if len(data['weights'])!=len(n.weights) or any(not math.isfinite(w) for w in data['weights']): raise ValueError('Invalid weight vector')
    n.weights=list(data['weights']); return n
