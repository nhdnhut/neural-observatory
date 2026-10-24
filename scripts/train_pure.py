"""End-to-end standard-library training. Demonstration subset, not full benchmark.

python -S scripts/train_pure.py --samples 1024 --epochs 2
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from observatory.dataset import download,read_images,read_labels,split_indices
from observatory.reference import ReferenceMLP
from observatory.packing import export_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--samples",type=int,default=1024)
    parser.add_argument("--epochs",type=int,default=2)
    parser.add_argument("--width",type=int,default=20)
    args=parser.parse_args()
    if not 1<=args.samples<=55000 or args.epochs<1 or args.width<1:
        parser.error("Invalid training size, epoch count or width")
    root=download(ROOT/"data/raw")
    raw,_=read_images(root/"train-images-idx3-ubyte.gz")
    labels=read_labels(root/"train-labels-idx1-ubyte.gz")
    train,val=split_indices(labels)
    train,val=train[:args.samples],val[:256]
    def image(i):
        return [v/255 for v in raw[784*i:784*(i+1)]]
    x=[image(i) for i in train]; y=[labels[i] for i in train]
    xv=[image(i) for i in val]; yv=[labels[i] for i in val]
    model=ReferenceMLP((784,args.width,10),"relu",42)
    rng=random.Random(42)
    history=[]
    start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        order=list(range(len(y)));rng.shuffle(order)
        losses=[model.train_one(x[i],y[i],.01) for i in order]
        acc=sum(model.predict(xx)==yy for xx,yy in zip(xv,yv))/len(yv)
        row={"epoch":epoch,"online_training_loss":sum(losses)/len(losses),"validation_accuracy":acc,"elapsed_s":time.perf_counter()-start}
        history.append(row);print(row,flush=True)
    output=ROOT/"artifacts/pure_python"
    output.mkdir(parents=True,exist_ok=True)
    export_model(output/"model.fp32.nobs",model.dims,model.activation,model.weights,model.biases)
    export_model(output/"model.int8.nobs",model.dims,model.activation,model.weights,model.biases,True)
    result={"runtime":"Python standard library; tested with python -S", "train_samples":len(y),"validation_samples":len(yv),
            "test_evaluated":False,"architecture":model.dims,"seed":42,"learning_rate":.01,"optimizer":"per-sample SGD",
            "history":history,"purpose":"algorithm demonstration, not comparable to the full NumPy benchmark"}
    (output/"results.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
