"""Standard-library inference from saved FP32 or INT8 weights."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from observatory.dataset import download,read_images,read_labels
from observatory.packing import load_packed,packed_forward


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default=str(ROOT/"artifacts/models/observatory_64_32_s42.int8.nobs"))
    p.add_argument("--index",type=int,default=0)
    args=p.parse_args()
    download(ROOT/"data/raw")
    raw,n=read_images(ROOT/"data/raw/t10k-images-idx3-ubyte.gz")
    if not 0<=args.index<n:
        p.error(f"Index must be in 0..{n-1}")
    y=read_labels(ROOT/"data/raw/t10k-labels-idx1-ubyte.gz")[args.index]
    x=[v/255 for v in raw[args.index*784:(args.index+1)*784]]
    model=load_packed(args.model)
    probabilities=packed_forward(model,x)
    print({"index":args.index,"label":y,"prediction":max(range(10),key=lambda c:probabilities[c]),"probabilities":probabilities})


if __name__=="__main__":
    main()
