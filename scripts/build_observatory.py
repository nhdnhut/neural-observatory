"""Analyze saved checkpoints; generate real-data visualizations and offline lab."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import base64
import csv
import hashlib
import json
import statistics
import numpy as np
from observatory.engine import MLP
from observatory.dataset import load_numpy
from observatory.packing import load_packed
from train_suite import unpack_for_evaluation,write_json


def encode_model(model):
    return {"dims":model.dims,"activation":model.activation,
            "weights":[base64.b64encode(w.astype('<f4').tobytes()).decode() for w in model.weights],
            "biases":[b.tolist() for b in model.biases]}


def aggregate(results):
    groups=[]
    for name in dict.fromkeys(r['model'] for r in results):
        for variant in ('fp32','int8'):
            rows=[r for r in results if r['model']==name and r['variant']==variant]
            groups.append({"model":name,"variant":variant,"dims":rows[0]['dims'],"parameters":rows[0]['parameters'],
                           "model_bytes":rows[0]['model_bytes'],"weight_bytes":rows[0]['packed_weight_buffer_bytes'],
                           "accuracy_mean":statistics.mean(r['test']['accuracy'] for r in rows),
                           "accuracy_sd":statistics.stdev(r['test']['accuracy'] for r in rows) if len(rows)>1 else 0.,
                           "validation_mean":statistics.mean(r['validation']['accuracy'] for r in rows),
                           "latency_p50_ms":statistics.median(r['latency']['p50_ms'] for r in rows),
                           "latency_p95_ms":statistics.median(r['latency']['p95_ms'] for r in rows),
                           "seeds":[r['seed'] for r in rows],"changes":[r['prediction_changes_vs_fp32'] for r in rows]})
    return groups


def neuron_ablation(model,x,y):
    acts,zs=model.forward(x)
    base=acts[-1]
    base_acc=float((base.argmax(axis=1)==y).mean())
    items=[]
    for layer in range(len(model.weights)-1):
        for neuron in range(model.dims[layer+1]):
            a=acts[layer+1].copy();a[:,neuron]=0
            for l in range(layer+1,len(model.weights)):
                z=a@model.weights[l].T+model.biases[l]
                if l==len(model.weights)-1:
                    e=np.exp(z-z.max(axis=1,keepdims=True));a=e/e.sum(axis=1,keepdims=True)
                else:
                    a=np.maximum(z,0) if model.activation=='relu' else 1/(1+np.exp(-z))
            items.append({"layer":layer,"neuron":neuron,
                          "accuracy_drop_pp":100*(base_acc-float((a.argmax(axis=1)==y).mean())),
                          "mean_true_probability_drop":float((base[np.arange(len(y)),y]-a[np.arange(len(y)),y]).mean())})
    return {"split":"first 1024 validation examples in the fixed split order","n":len(y),"baseline_accuracy":base_acc,"items":items}


def figures(manifest,groups,model,x_test,y_test,ablation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'axes.titlesize':13,'axes.titleweight':'bold','axes.labelsize':10,'figure.facecolor':'#f6f7fb','axes.facecolor':'#ffffff'})
    violet='#6457d9';teal='#087e82';orange='#d97835'
    main=next(r for r in manifest['results'] if r['model']=='observatory_64_32' and r['seed']==42 and r['variant']=='fp32')
    h=main['history'];ep=[r['epoch'] for r in h]
    fig,axs=plt.subplots(2,2,figsize=(13.5,8.8),layout='constrained')
    ax=axs[0,0]
    ax.plot(ep,[r['train_accuracy']*100 for r in h],color=violet,label='Train diagnostic subset (5,000)')
    ax.plot(ep,[r['val_accuracy']*100 for r in h],color=teal,label='Validation (5,000)')
    ax.set(xlabel='Epoch',ylabel='Accuracy (%)',title='Learning measured at every epoch');ax.legend(fontsize=8);ax.grid(alpha=.15)
    ax=axs[0,1]
    ax.plot(ep,[r['train_loss'] for r in h],color=violet,label='Train subset')
    ax.plot(ep,[r['val_loss'] for r in h],color=teal,label='Validation')
    ax.set(xlabel='Epoch',ylabel='Cross-entropy',title='Loss and generalization gap');ax.legend(fontsize=8);ax.grid(alpha=.15)
    ax=axs[1,0]
    for g in groups:
        color=violet if g['variant']=='fp32' else teal
        ax.errorbar(g['model_bytes']/1024,g['accuracy_mean']*100,yerr=g['accuracy_sd']*100,
                    marker='o' if g['variant']=='fp32' else 's',color=color,capsize=3,linestyle='none')
        label={'observatory_64_32':'64-32','baseline_20_sigmoid':'20 sigmoid','compact_20_relu':'20 ReLU','budget_6_relu':'6 ReLU'}[g['model']]
        offset=(5,7 if g['model']!='baseline_20_sigmoid' else -14)
        align='left'
        if g['model']=='budget_6_relu':
            offset=(6,-20 if g['variant']=='int8' else 10)
        if g['model']=='observatory_64_32' and g['variant']=='fp32':
            offset=(-7,8);align='right'
        ax.annotate(label+' '+g['variant'].upper(),(g['model_bytes']/1024,g['accuracy_mean']*100),
                    xytext=offset,textcoords='offset points',fontsize=7,ha=align)
    ax.set(xlabel='Serialized model size (KiB)',ylabel='Test accuracy (%)',title='Accuracy vs. model storage: 3 seeds');ax.grid(alpha=.15)
    cm=np.array(main['confusion']);ax=axs[1,1]
    im=ax.imshow(cm,cmap='Purples')
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,color='white' if cm[i,j]>500 else '#252040')
    ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted digit',ylabel='True digit',title='Held-out test confusion matrix: seed 42')
    fig.suptitle('NEURAL OBSERVATORY  /  Actual MNIST experiment',fontsize=20,fontweight='bold',x=.05,ha='left')
    fig.savefig(ROOT/'artifacts/figures/experiment_overview.png',dpi=170);fig.savefig(ROOT/'artifacts/figures/experiment_overview.svg');plt.close(fig)
    weights=model.weights[0].reshape(64,28,28)
    vmax=float(np.abs(weights).max())
    fig,axs=plt.subplots(8,8,figsize=(10,10),layout='constrained')
    for j,ax in enumerate(axs.flat):
        ax.imshow(weights[j],cmap='RdBu_r',vmin=-vmax,vmax=vmax);ax.set_title(f'N{j+1}',fontsize=8);ax.axis('off')
    fig.suptitle('Layer 1: learned input weights, 64 neurons\nShared symmetric color scale; weights are not semantic explanations',fontsize=13)
    fig.savefig(ROOT/'artifacts/figures/learned_weights.png',dpi=150);plt.close(fig)
    selected=sorted(ablation['items'],key=lambda r:r['accuracy_drop_pp'],reverse=True)[:15]
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    ax.bar([f"L{r['layer']+1}/N{r['neuron']+1}" for r in selected],[r['accuracy_drop_pp'] for r in selected],color=teal)
    ax.set(ylabel='Accuracy drop (percentage points)',xlabel='Ablated neuron',title='Single-neuron interventions on 1,024 validation examples')
    ax.tick_params(axis='x',labelrotation=40)
    fig.savefig(ROOT/'artifacts/figures/neuron_ablation.png',dpi=150);plt.close(fig)


def main():
    manifest=json.loads((ROOT/'artifacts/experiment.json').read_text())
    x,y,xv,yv,xt,yt,_,_=load_numpy(ROOT/'data/raw')
    # Recompute the validation score of the exact serialized variant. This also
    # makes the aggregate budget selection use quantized validation scores.
    for r in manifest['results']:
        restored=unpack_for_evaluation(load_packed(ROOT/r['path']))
        r['train']=restored.evaluate(x,y)
        r['validation']=restored.evaluate(xv,yv)
    write_json(ROOT/'artifacts/experiment.json',manifest)
    for name,seed in dict.fromkeys((r['model'],r['seed']) for r in manifest['results']):
        rows=[r for r in manifest['results'] if r['model']==name and r['seed']==seed]
        write_json(ROOT/'artifacts/checkpoints'/f'{name}_s{seed}'/'results.json',rows)
    groups=aggregate(manifest['results'])
    write_json(ROOT/'artifacts/aggregate.json',groups)
    main_result=next(r for r in manifest['results'] if r['model']=='observatory_64_32' and r['seed']==42 and r['variant']=='fp32')
    run=ROOT/'artifacts/checkpoints/observatory_64_32_s42'
    model=MLP.load(run/'best.npz')
    qmodel=unpack_for_evaluation(load_packed(ROOT/'artifacts/models/observatory_64_32_s42.int8.nobs'))
    probs=model.forward(xt)[0][-1];pred=probs.argmax(axis=1)
    qprobs=qmodel.forward(xt)[0][-1];qpred=qprobs.argmax(axis=1)
    chosen=[]
    for digit in range(10):
        # Deterministic representative correct samples, not a performance estimate.
        chosen.extend(np.where((yt==digit)&(pred==yt))[0][:4].tolist())
    errors=np.where(pred!=yt)[0]
    chosen.extend(errors[:40].tolist())
    changes=np.where(qpred!=pred)[0]
    chosen.extend(changes[:30].tolist())
    chosen=list(dict.fromkeys(chosen))
    samples=[]
    for i in chosen:
        samples.append({"index":int(i),"label":int(yt[i]),"prediction":int(pred[i]),"quantized_prediction":int(qpred[i]),
                        "error":bool(pred[i]!=yt[i]),"quantization_changed":bool(qpred[i]!=pred[i]),
                        "pixels":base64.b64encode(np.rint(xt[i]*255).astype(np.uint8).tobytes()).decode(),
                        "reference_probabilities":probs[i].tolist(),"quantized_probabilities":qprobs[i].tolist()})
    selected_epochs=sorted(set([0,1,2,4,8,12,18,main_result['best_epoch']]))
    selected_epochs=[e for e in selected_epochs if (run/f'epoch_{e:03d}.npz').exists()]
    checkpoints=[{"epoch":e,**encode_model(MLP.load(run/f'epoch_{e:03d}.npz'))} for e in selected_epochs]
    ablation=neuron_ablation(model,xv[:1024],yv[:1024])
    write_json(ROOT/'artifacts/neuron_ablation.json',ablation)
    # Fixed mild translation: zero-fill 2 pixels right, no wrapping. Diagnostic
    # held-out corruption, not used for model or checkpoint selection.
    shifted=np.zeros_like(xt.reshape(-1,28,28));shifted[:,:,2:]=xt.reshape(-1,28,28)[:,:,:-2]
    robustness={"transform":"zero-filled translation 2 pixels right","split":"official test (10000)",
                "clean":main_result['test'],"shifted":model.evaluate(shifted.reshape(-1,784),yt)}
    write_json(ROOT/'artifacts/robustness.json',robustness)
    packed=load_packed(ROOT/'artifacts/models/observatory_64_32_s42.int8.nobs')
    qpayload={"dims":packed['dims'],"activation":packed['activation'],"quantized":True,
              "weights":[base64.b64encode(w.tobytes()).decode() for w in packed['weights']],
              "biases":packed['biases'],"scales":packed['scales']}
    payload={"title":"Neural Observatory","author":"Nguyễn Huỳnh Đăng Nhựt · 26521522",
             "problem":"Tự xây dựng, quan sát và nén mạng neural dưới ngân sách bộ nhớ",
             "protocol":{k:v for k,v in manifest.items() if k!='results'},"main":main_result,"groups":groups,
             "history":main_result['history'],"checkpoints":checkpoints,"best_epoch":main_result['best_epoch'],
             "quantized":qpayload,"samples":samples,"ablation":ablation,"robustness":robustness,
             "pure_python":json.loads((ROOT/'artifacts/pure_python/results.json').read_text())}
    # Browser data is a lossless binary representation of the trained weights.
    (ROOT/'dashboard/data.js').write_text('window.OBSERVATORY_DATA='+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+';\n')
    figures(manifest,groups,model,xt,yt,ablation)
    template=(ROOT/'dashboard/index.html').read_text()
    standalone=template.replace('<link rel="stylesheet" href="styles.css">','<style>'+(ROOT/'dashboard/styles.css').read_text()+'</style>')
    standalone=standalone.replace('<script src="data.js"></script>','<script>'+(ROOT/'dashboard/data.js').read_text()+'</script>')
    standalone=standalone.replace('<script src="app.js"></script>','<script>'+(ROOT/'dashboard/app.js').read_text()+'</script>')
    (ROOT/'artifacts/Neural_Observatory.html').write_text(standalone)
    print('Built interactive lab and figures. Test seed42:',main_result['test'],'Samples:',len(samples),'Checkpoints:',selected_epochs)


if __name__=='__main__':
    main()
