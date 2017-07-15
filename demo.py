import json
from core import Network
from features import gradient_error
with open('examples/input.json') as stream: data=json.load(stream)
n=Network(seed=data['seed']); error=gradient_error(n,data['rows'][0],data['targets'][0])
n.train(data['rows'],data['targets'],epochs=6000)
print(json.dumps({'gradient_error':error,'predictions':[round(n.predict(r),4) for r in data['rows']]},sort_keys=True))
