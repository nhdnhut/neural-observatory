"""Batched manual backpropagation. NumPy performs array arithmetic only."""
import json
import numpy as np


class MLP:
    def __init__(self, dims=(784,64,32,10), activation="relu", seed=42, dtype=np.float32):
        if activation not in ("relu", "sigmoid"):
            raise ValueError("Unknown activation")
        self.dims, self.activation = list(dims), activation
        rng = np.random.default_rng(seed)
        self.weights, self.biases = [], []
        for l, (nin,nout) in enumerate(zip(dims[:-1],dims[1:])):
            std = np.sqrt((2.0 if activation=="relu" and l < len(dims)-2 else 1.0)/nin)
            self.weights.append(rng.normal(0,std,(nout,nin)).astype(dtype))
            self.biases.append(np.zeros(nout,dtype=dtype))
        self.vw = [np.zeros_like(w) for w in self.weights]
        self.vb = [np.zeros_like(b) for b in self.biases]

    @property
    def parameters(self):
        return sum(w.size+b.size for w,b in zip(self.weights,self.biases))

    def forward(self, x, intervention=None):
        acts, zs = [np.atleast_2d(x)], []
        for l,(w,b) in enumerate(zip(self.weights,self.biases)):
            z = acts[-1] @ w.T + b
            zs.append(z)
            if l == len(self.weights)-1:
                e = np.exp(z-z.max(axis=1,keepdims=True))
                a = e / e.sum(axis=1,keepdims=True)
            elif self.activation == "relu":
                a = np.maximum(z,0)
            else:
                a = 1/(1+np.exp(-np.clip(z,-80,80)))
            if intervention is not None and intervention[0] == l:
                if l == len(self.weights)-1:
                    raise ValueError("Intervene on hidden layers only")
                a = a.copy()
                a[:,intervention[1]] = 0
            acts.append(a)
        return acts,zs

    def loss_and_grad(self, x, y):
        acts,zs = self.forward(x)
        logits = zs[-1]
        maximum = logits.max(axis=1,keepdims=True)
        losses = np.log(np.exp(logits-maximum).sum(axis=1)) + maximum[:,0] - logits[np.arange(len(y)),y]
        delta = acts[-1].copy()
        delta[np.arange(len(y)),y] -= 1
        delta /= len(y)
        gw,gb = [None]*len(self.weights),[None]*len(self.weights)
        for l in reversed(range(len(self.weights))):
            gw[l] = delta.T @ acts[l]
            gb[l] = delta.sum(axis=0)
            if l:
                delta = delta @ self.weights[l]
                delta *= zs[l-1]>0 if self.activation=="relu" else acts[l]*(1-acts[l])
        return float(losses.mean()),gw,gb

    def step(self, gw, gb, lr=0.03, momentum=0.9):
        for l in range(len(self.weights)):
            self.vw[l] *= momentum
            self.vw[l] += gw[l]
            self.vb[l] *= momentum
            self.vb[l] += gb[l]
            self.weights[l] -= lr*self.vw[l]
            self.biases[l] -= lr*self.vb[l]

    def evaluate(self, x, y, batch_size=1024):
        loss, correct = 0.0, 0
        for start in range(0,len(y),batch_size):
            xx,yy = x[start:start+batch_size],y[start:start+batch_size]
            acts,zs = self.forward(xx)
            zz = zs[-1]
            mm = zz.max(axis=1,keepdims=True)
            loss += float((np.log(np.exp(zz-mm).sum(axis=1))+mm[:,0]-zz[np.arange(len(yy)),yy]).sum())
            correct += int((acts[-1].argmax(axis=1)==yy).sum())
        return {"loss":loss/len(y),"accuracy":correct/len(y)}

    def save(self, path):
        arrays={"meta":np.array(json.dumps({"dims":self.dims,"activation":self.activation}))}
        for l,(w,b) in enumerate(zip(self.weights,self.biases)):
            arrays[f"w{l}"]=w
            arrays[f"b{l}"]=b
        np.savez_compressed(path,**arrays)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data["meta"]))
            model=cls(meta["dims"],meta["activation"])
            for l in range(len(model.weights)):
                model.weights[l]=data[f"w{l}"].copy()
                model.biases[l]=data[f"b{l}"].copy()
        return model

    def layer_stats(self,x):
        acts,zs=self.forward(x)
        return [{"layer":l+1,"width":a.shape[1],"mean":float(a.mean()),"std":float(a.std()),
                 "zero_fraction":float((a==0).mean()),"inactive_neurons":int((a.max(axis=0)==0).sum())}
                for l,a in enumerate(acts[1:-1])]
