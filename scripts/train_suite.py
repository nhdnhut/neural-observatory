"""Reproducible CPU study. Run from repository root: python scripts/train_suite.py.

All model/optimizer choices below are fixed before inspecting the test set.
Only validation cross-entropy selects the checkpoint within each run.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime,timezone
import numpy as np
from observatory.dataset import load_numpy,FILES
from observatory.engine import MLP
from observatory.packing import export_model,load_packed,packed_forward

CONFIGS=[
    {"id":"observatory_64_32","dims":[784,64,32,10],"activation":"relu","lr":0.04},
    {"id":"baseline_20_sigmoid","dims":[784,20,10],"activation":"sigmoid","lr":0.12},
    {"id":"compact_20_relu","dims":[784,20,10],"activation":"relu","lr":0.04},
    {"id":"budget_6_relu","dims":[784,6,10],"activation":"relu","lr":0.04},
]


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":",1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def write_json(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))


def unpack_for_evaluation(packed):
    """Dequantized mathematical evaluation, NOT an integer-only runtime."""
    model=MLP(packed["dims"],packed["activation"])
    for l,(nin,nout) in enumerate(zip(model.dims[:-1],model.dims[1:])):
        model.weights[l]=np.asarray(packed["weights"][l],dtype=np.float32).reshape(nout,nin)*np.asarray(packed["scales"][l],dtype=np.float32)[:,None]
        model.biases[l]=np.asarray(packed["biases"][l],dtype=np.float32)
    return model


def latency(packed,x,n=100,repeats=3):
    samples=[row.tolist() for row in x[:n]]
    for row in samples[:10]:
        packed_forward(packed,row)
    measurements=[]
    for _ in range(repeats):
        for row in samples:
            start=time.perf_counter_ns()
            packed_forward(packed,row)
            measurements.append((time.perf_counter_ns()-start)/1e6)
    return {"runtime":"standard_library_packed_forward_float_arithmetic","batch_size":1,
            "n_samples":n,"repeats":repeats,"warmup_calls":10,
            "p50_ms":statistics.median(measurements),"p95_ms":float(np.percentile(measurements,95)),
            "measurements_ms":measurements}


def confusion(model,x,y):
    pred=model.forward(x)[0][-1].argmax(axis=1)
    cm=np.zeros((10,10),dtype=int)
    np.add.at(cm,(y,pred),1)
    return cm.tolist(),pred


def train_run(config,seed,data,args):
    x,y,xv,yv,xt,yt,_,_=data
    name=f'{config["id"]}_s{seed}'
    run_dir=ROOT/"artifacts"/"checkpoints"/name
    run_dir.mkdir(parents=True,exist_ok=True)
    model=MLP(config["dims"],config["activation"],seed)
    rng=np.random.default_rng(seed+1000)
    start=time.perf_counter()
    initial=model.evaluate(xv,yv)
    # Same fixed training subset for epoch-level diagnostic curves, all runs.
    initial_train=model.evaluate(x[:5000],y[:5000])
    history=[{"epoch":0,"train_loss":initial_train["loss"],"train_accuracy":initial_train["accuracy"],
              "val_loss":initial["loss"],"val_accuracy":initial["accuracy"],"lr":config["lr"],
              "elapsed_s":0.,"layer_stats":model.layer_stats(xv[:1024]),"gradient_norm":[]}]
    if seed==args.seeds[0]:
        model.save(run_dir/"epoch_000.npz")
    best_loss=float("inf")
    best_epoch=0
    for epoch in range(1,args.epochs+1):
        order=rng.permutation(len(y))
        lr=config["lr"]*(0.1+0.9*(1+np.cos(np.pi*(epoch-1)/max(1,args.epochs-1)))/2)
        norms=[]
        for step,begin in enumerate(range(0,len(y),args.batch_size)):
            indices=order[begin:begin+args.batch_size]
            _,gw,gb=model.loss_and_grad(x[indices],y[indices])
            if step==0:
                norms=[float(np.linalg.norm(g)) for g in gw]
            model.step(gw,gb,float(lr),momentum=0.9)
        val=model.evaluate(xv,yv)
        train=model.evaluate(x[:5000],y[:5000])
        history.append({"epoch":epoch,"train_loss":train["loss"],"train_accuracy":train["accuracy"],
                        "val_loss":val["loss"],"val_accuracy":val["accuracy"],"lr":float(lr),
                        "elapsed_s":time.perf_counter()-start,"layer_stats":model.layer_stats(xv[:1024]),"gradient_norm":norms})
        if val["loss"] < best_loss:
            best_loss,best_epoch=val["loss"],epoch
            model.save(run_dir/"best.npz")
        if seed==args.seeds[0]:
            model.save(run_dir/f"epoch_{epoch:03d}.npz")
        print(f"{name} epoch {epoch:02d}/{args.epochs}: train={train['accuracy']:.4f} val={val['accuracy']:.4f} loss={val['loss']:.4f}",flush=True)
        write_json(run_dir/"history.json",history)
    training_seconds=time.perf_counter()-start
    model=MLP.load(run_dir/"best.npz")
    train_all=model.evaluate(x,y)
    selected_validation=model.evaluate(xv,yv)
    results=[]
    fp_predictions=None
    for quantized in (False,True):
        variant="int8" if quantized else "fp32"
        path=ROOT/"artifacts"/"models"/f"{name}.{variant}.nobs"
        size=export_model(path,model.dims,model.activation,[w.tolist() for w in model.weights],
                          [b.tolist() for b in model.biases],quantized)
        packed=load_packed(path)
        evaluated=unpack_for_evaluation(packed)
        test=evaluated.evaluate(xt,yt)
        cm,predictions=confusion(evaluated,xt,yt)
        if fp_predictions is None:
            fp_predictions=predictions
        timing=latency(packed,xt,args.latency_samples)
        # Array payload counts; interpreter/object overhead is excluded and labeled.
        weight_bytes=sum(len(w)*w.itemsize for w in packed["weights"])
        result={"model":config["id"],"seed":seed,"variant":variant,"dims":model.dims,
                "activation":model.activation,"parameters":model.parameters,"best_epoch":best_epoch,
                "train":evaluated.evaluate(x,y),"validation":evaluated.evaluate(xv,yv),"checkpoint_validation_fp32":selected_validation,"test":test,
                "model_bytes":size,"packed_weight_buffer_bytes":weight_bytes,
                "prediction_changes_vs_fp32":int((predictions!=fp_predictions).sum()),
                "training_seconds":training_seconds,"latency":timing,"confusion":cm,
                "model_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "path":str(path.relative_to(ROOT)),"history":history}
        results.append(result)
        print(f"RESULT {name} {variant}: test={test['accuracy']:.4f}; bytes={size}; p50={timing['p50_ms']:.3f}ms",flush=True)
    write_json(run_dir/"results.json",results)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=18)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    parser.add_argument("--latency-samples",type=int,default=100)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.latency_samples<1:
        parser.error("Positive epochs/batch/latency sample count required")
    os.chdir(ROOT)
    data=load_numpy()
    manifest={"project":"Neural Observatory","source_repository":"https://github.com/DangNhutNguyen/Neural-Network-from-Scratch",
              "source_commit":"c3ce89b4cc22e034611287e30af1b5dda6d17bbe",
              "started_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
              "numpy":np.__version__,"platform":platform.platform(),"cpu":cpu_name(),"threads":1,
              "dataset":{"name":"MNIST","checksums_sha256":FILES,"train":len(data[1]),"validation":len(data[3]),
                         "test":len(data[5]),"split_seed":2026,"validation_per_class":500,
                         "train_indices_sha256":hashlib.sha256(np.array(data[6],dtype='<i8').tobytes()).hexdigest(),
                         "validation_indices_sha256":hashlib.sha256(np.array(data[7],dtype='<i8').tobytes()).hexdigest()},
              "protocol":{"epochs":args.epochs,"seeds":args.seeds,"batch_size":args.batch_size,"optimizer":"manual SGD momentum 0.9",
                          "checkpoint_selection":"minimum validation cross-entropy",
                          "training_curve_subset":5000,"test_used_for_selection":False,
                          "latency_samples":args.latency_samples,"latency_repeats":3,"configs":CONFIGS},
              "results":[]}
    write_json(ROOT/"artifacts"/"protocol.json",{k:v for k,v in manifest.items() if k!="results"})
    for config in CONFIGS:
        for seed in args.seeds:
            prior=ROOT/"artifacts"/"checkpoints"/f'{config["id"]}_s{seed}'/"results.json"
            if args.resume and prior.exists():
                result=json.loads(prior.read_text())
                if len(result[0]["history"])!=args.epochs+1:
                    raise ValueError("Resume epoch mismatch")
            else:
                result=train_run(config,seed,data,args)
            manifest["results"].extend(result)
            write_json(ROOT/"artifacts"/"experiment.json",manifest)
    rows=[]
    for r in manifest["results"]:
        rows.append({"model":r["model"],"seed":r["seed"],"variant":r["variant"],"parameters":r["parameters"],
                     "best_epoch":r["best_epoch"],"test_accuracy":r["test"]["accuracy"],"test_loss":r["test"]["loss"],
                     "model_bytes":r["model_bytes"],"packed_weight_buffer_bytes":r["packed_weight_buffer_bytes"],
                     "latency_p50_ms":r["latency"]["p50_ms"],"latency_p95_ms":r["latency"]["p95_ms"],
                     "prediction_changes_vs_fp32":r["prediction_changes_vs_fp32"],"training_seconds":r["training_seconds"]})
    with open(ROOT/"artifacts"/"benchmark.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    print("Saved artifacts/experiment.json and artifacts/benchmark.csv",flush=True)


if __name__=="__main__":
    main()
