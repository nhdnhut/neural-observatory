import math
import tempfile
import unittest
from pathlib import Path
from observatory.reference import ReferenceMLP,softmax
from observatory.packing import quantize_row,export_model,load_packed,packed_forward


class AlgorithmTests(unittest.TestCase):
    def test_softmax_extremes(self):
        p=softmax([10000,10001,-10000])
        self.assertAlmostEqual(sum(p),1.0)
        self.assertTrue(all(math.isfinite(v) for v in p))

    def test_all_reference_gradients_finite_difference(self):
        for act in ("sigmoid","relu"):
            m=ReferenceMLP((3,4,2),act,42)
            x=[.3,-.7,.4]
            _,gw,gb=m.loss_and_grad(x,1)
            eps=1e-5
            for l in range(len(m.weights)):
                for j,row in enumerate(m.weights[l]):
                    for i in range(len(row)):
                        old=row[i]
                        row[i]=old+eps
                        plus=m.loss_and_grad(x,1)[0]
                        row[i]=old-eps
                        minus=m.loss_and_grad(x,1)[0]
                        row[i]=old
                        self.assertAlmostEqual((plus-minus)/(2*eps),gw[l][j][i],places=7)
                for j in range(len(m.biases[l])):
                    old=m.biases[l][j]
                    m.biases[l][j]=old+eps
                    plus=m.loss_and_grad(x,1)[0]
                    m.biases[l][j]=old-eps
                    minus=m.loss_and_grad(x,1)[0]
                    m.biases[l][j]=old
                    self.assertAlmostEqual((plus-minus)/(2*eps),gb[l][j],places=7)

    def test_quantization_error_bound_and_zero(self):
        q,s=quantize_row([0.,0.,0.])
        self.assertEqual(q,[0,0,0])
        self.assertEqual(s,1.)
        values=[-2.,-.13,0.,.32,1.1]
        q,s=quantize_row(values)
        for v,qq in zip(values,q):
            self.assertLessEqual(abs(v-qq*s),s/2+1e-6)

    def test_packed_roundtrip_predictions(self):
        m=ReferenceMLP((3,8,2),"sigmoid")
        x=[.2,.8,-.1]
        p=m.forward(x)[0][-1]
        with tempfile.TemporaryDirectory() as d:
            for quantized in (False,True):
                path=Path(d)/"model.bin"
                export_model(path,m.dims,m.activation,m.weights,m.biases,quantized)
                pp=packed_forward(load_packed(path),x)
                for a,b in zip(p,pp):
                    self.assertAlmostEqual(a,b,places=2 if quantized else 6)

    def test_numpy_matches_reference_forward_and_gradients(self):
        try:
            import numpy as np
            from observatory.engine import MLP
        except ImportError:
            self.skipTest("Optional NumPy backend not installed")
        for act in ("relu","sigmoid"):
            r=ReferenceMLP((3,4,3,2),act,12)
            n=MLP(r.dims,act,dtype=np.float64)
            n.weights=[np.asarray(w) for w in r.weights]
            n.biases=[np.asarray(b) for b in r.biases]
            x,y=[.2,-.7,.8],1
            rl,rw,rb=r.loss_and_grad(x,y)
            nl,nw,nb=n.loss_and_grad(np.array([x]),np.array([y]))
            self.assertAlmostEqual(rl,nl,places=12)
            for a,b in zip(rw,nw):
                np.testing.assert_allclose(a,b,atol=1e-12)
            for a,b in zip(rb,nb):
                np.testing.assert_allclose(a,b,atol=1e-12)


if __name__=="__main__":
    unittest.main()
