"""Pure-standard-library FP32 / symmetric per-output-channel INT8 storage.

This is WEIGHT-ONLY quantization. Biases and scales are float32; arithmetic
is floating point. It does not claim integer-only kernels or faster inference.
"""
from array import array
import json
import math
import struct
import sys

MAGIC32=b"NOBS32v1"
MAGIC8=b"NOBS08v1"


def quantize_row(row):
    if not row or not all(math.isfinite(w) for w in row):
        raise ValueError("Weights must be finite and nonempty")
    max_abs=max(abs(w) for w in row)
    scale=max_abs/127 if max_abs else 1.0
    # Quantizer uses a representable float32 scale, same value stored on disk.
    scale=struct.unpack("<f",struct.pack("<f",scale))[0]
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("Scale outside representable float32 range")
    q=[max(-127,min(127,round(w/scale))) for w in row]
    return q,scale


def export_model(path, dims, activation, weights, biases, quantized=False):
    meta=json.dumps({"dims":list(dims),"activation":activation,"quantization":"per_output_channel_symmetric_int8" if quantized else "none",
                     "arithmetic":"floating_point","bias_dtype":"float32"},separators=(",",":")).encode()
    out=bytearray(MAGIC8 if quantized else MAGIC32)
    out.extend(struct.pack("<I",len(meta)))
    out.extend(meta)
    for w,b in zip(weights,biases):
        if quantized:
            rows,scales=[],[]
            for row in w:
                q,s=quantize_row(row)
                rows.extend(q)
                scales.append(s)
            out.extend(struct.pack("<"+str(len(rows))+"b",*rows))
            out.extend(struct.pack("<"+str(len(scales))+"f",*scales))
        else:
            flat=[float(x) for row in w for x in row]
            out.extend(struct.pack("<"+str(len(flat))+"f",*flat))
        out.extend(struct.pack("<"+str(len(b))+"f",*b))
    with open(path,"wb") as f:
        f.write(out)
    return len(out)


def load_packed(path):
    with open(path,"rb") as f:
        raw=f.read()
    magic=raw[:8]
    if magic not in (MAGIC32,MAGIC8):
        raise ValueError("Invalid model magic")
    nmeta=struct.unpack_from("<I",raw,8)[0]
    meta=json.loads(raw[12:12+nmeta])
    offset=12+nmeta
    result={**meta,"quantized":magic==MAGIC8,"weights":[],"biases":[],"scales":[]}
    for nin,nout in zip(meta["dims"][:-1],meta["dims"][1:]):
        count=nin*nout
        width=1 if magic==MAGIC8 else 4
        values=array("b" if magic==MAGIC8 else "f")
        values.frombytes(raw[offset:offset+count*width])
        if width==4 and sys.byteorder!="little":
            values.byteswap()
        if len(values)!=count:
            raise ValueError("Truncated model")
        offset+=count*width
        scales=[1.0]*nout
        if magic==MAGIC8:
            scales=list(struct.unpack_from("<"+str(nout)+"f",raw,offset))
            offset+=nout*4
        biases=list(struct.unpack_from("<"+str(nout)+"f",raw,offset))
        offset+=nout*4
        result["weights"].append(values)
        result["biases"].append(biases)
        result["scales"].append(scales)
    if offset!=len(raw):
        raise ValueError("Unexpected trailing model bytes")
    return result


def packed_forward(model,x):
    """Inference without materializing a full floating-point weight matrix."""
    from .reference import sigmoid,softmax
    a=list(x)
    for l,(nin,nout) in enumerate(zip(model["dims"][:-1],model["dims"][1:])):
        w,b,s=model["weights"][l],model["biases"][l],model["scales"][l]
        z=[sum(w[j*nin+i]*a[i] for i in range(nin))*s[j]+b[j] for j in range(nout)]
        if l==len(model["weights"])-1:
            a=softmax(z)
        else:
            a=[max(0.,v) for v in z] if model["activation"]=="relu" else [sigmoid(v) for v in z]
    return a
