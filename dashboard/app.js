/* Neural Observatory. Framework-free numerical inference and visual inspection.
   Stored weights come from Python training. No training is simulated here. */
(() => {
  'use strict';
  const D = window.OBSERVATORY_DATA;
  if (!D) throw new Error('Missing measured experiment data. Run scripts/build_observatory.py.');
  const $ = id => document.getElementById(id);
  const colors = ['#6653ce','#087e82','#c76528','#5684b9'];
  const names = {observatory_64_32:'Observatory · 64–32',baseline_20_sigmoid:'Đối chứng · 20 Sigmoid',compact_20_relu:'Compact · 20 ReLU',budget_6_relu:'Budget · 6 ReLU'};
  const pct = v => (100*v).toFixed(2)+'%';
  const bytes = b => (b/1024).toFixed(2)+' KiB';
  function binary(s) {const b=atob(s);return Uint8Array.from(b,c=>c.charCodeAt(0));}
  function decode(m) {
    const weights=m.weights.map(s=>{
      const u=binary(s);
      if(m.quantized) return new Int8Array(u.buffer);
      const dv=new DataView(u.buffer),v=new Float32Array(u.length/4);
      for(let i=0;i<v.length;i++)v[i]=dv.getFloat32(i*4,true);
      return v;
    });
    return {...m,weights};
  }
  const checkpoints=D.checkpoints.map(decode), quantized=decode(D.quantized);
  let epochIndex=checkpoints.findIndex(m=>m.epoch===D.best_epoch);
  if(epochIndex<0)epochIndex=checkpoints.length-1;
  let sampleIndex=0,filter='all',drawing=false,selectedLayer=0,selectedNeuron=5;
  let input=new Float64Array(784),disabled=[new Set(),new Set()],currentTab='inspect',isDrawing=false;
  function forward(m,x,off=[new Set(),new Set()]) {
    const acts=[Array.from(x)],zs=[];
    for(let l=0;l<m.weights.length;l++){
      const nin=m.dims[l],nout=m.dims[l+1],w=m.weights[l],b=m.biases[l];
      const z=new Array(nout);
      for(let j=0;j<nout;j++){
        let s=0;for(let i=0;i<nin;i++)s+=w[j*nin+i]*acts[l][i];
        z[j]=s*(m.quantized?m.scales[l][j]:1)+b[j];
      }
      zs.push(z);
      let a;
      if(l===m.weights.length-1){const mx=Math.max(...z);a=z.map(v=>Math.exp(v-mx));const s=a.reduce((p,v)=>p+v,0);a=a.map(v=>v/s);}
      else {a=z.map(v=>m.activation==='relu'?Math.max(0,v):1/(1+Math.exp(-v)));for(const j of off[l]||[])a[j]=0;}
      acts.push(a);
    }
    return {acts,zs,p:acts[acts.length-1]};
  }
  function argmax(a){return a.reduce((best,v,i)=>v>a[best]?i:best,0);}
  function saliency(m,trace,target,off){
    let grad=new Array(m.dims[m.dims.length-1]).fill(0);grad[target]=1;
    for(let l=m.weights.length-1;l>=0;l--){
      const prev=new Array(m.dims[l]).fill(0),w=m.weights[l];
      for(let j=0;j<m.dims[l+1];j++)for(let i=0;i<m.dims[l];i++)prev[i]+=w[j*m.dims[l]+i]*(m.quantized?m.scales[l][j]:1)*grad[j];
      if(l>0)for(let i=0;i<prev.length;i++){
        prev[i]*=m.activation==='relu'?(trace.zs[l-1][i]>0?1:0):trace.acts[l][i]*(1-trace.acts[l][i]);
        if(off[l-1]?.has(i))prev[i]=0;
      }
      grad=prev;
    }
    return grad.map(Math.abs);
  }
  function model(){return $('precision').value==='int8'?quantized:checkpoints[epochIndex];}
  function sampleList(){return D.samples.filter(s=>filter==='all'||(filter==='errors'&&s.error)||(filter==='correct'&&!s.error)||(filter==='changed'&&s.quantization_changed));}
  function paintPixels(canvas,values,cols=28,kind='gray'){
    const ctx=canvas.getContext('2d'),rows=Math.ceil(values.length/cols),cw=canvas.width/cols,ch=canvas.height/rows;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    const mx=kind==='signed'?Math.max(...values.map(Math.abs),1e-12):Math.max(...values,1e-12);
    values.forEach((v,i)=>{
      if(kind==='gray'){const n=Math.round(Math.max(0,Math.min(1,v))*255);ctx.fillStyle=`rgb(${n},${n},${n})`;}
      else if(kind==='signed'){
        const t=Math.min(1,Math.abs(v)/mx),base=v>=0?[202,91,52]:[69,103,183];
        ctx.fillStyle=`rgb(${base.map(c=>Math.round(246*(1-t)+c*t)).join(',')})`;
      } else {const t=Math.sqrt(v/mx);ctx.fillStyle=`rgb(${Math.round(20+230*t)},${Math.round(29+115*t)},${Math.round(43+5*t)})`;}
      ctx.fillRect((i%cols)*cw,Math.floor(i/cols)*ch,Math.ceil(cw),Math.ceil(ch));
    });
  }
  function paintActivations(layer,a){
    const c=$('layer'+layer+'Canvas'),ctx=c.getContext('2d'),cols=8,cell=30,rows=a.length/cols,oy=(240-rows*cell)/2;
    ctx.clearRect(0,0,240,240);const max=Math.max(...a,1e-12);
    a.forEach((v,i)=>{
      const x=(i%8)*cell+3,y=Math.floor(i/8)*cell+oy+3,t=Math.sqrt(v/max);
      ctx.fillStyle=`rgb(${Math.round(236-134*t)},${Math.round(233-150*t)},${Math.round(246-40*t)})`;
      ctx.beginPath();ctx.roundRect(x,y,24,24,4);ctx.fill();
      ctx.font='9px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle=t>.55?'#fff':'#7d749a';ctx.fillText(String(i+1),x+12,y+12);
      if(disabled[layer].has(i)){
        ctx.strokeStyle='#c65335';ctx.lineWidth=1.6;ctx.beginPath();ctx.moveTo(x+4,y+4);ctx.lineTo(x+20,y+20);ctx.moveTo(x+20,y+4);ctx.lineTo(x+4,y+20);ctx.stroke();
      }
      if(layer===selectedLayer&&i===selectedNeuron){ctx.strokeStyle='#172f39';ctx.lineWidth=2;ctx.strokeRect(x-1,y-1,26,26);}
    });
    $('active'+layer).textContent=a.filter(v=>v>0).length+'/'+a.length+' neuron hoạt động';
  }
  function selectSample(){
    drawing=false;document.body.classList.remove('draw-active');
    ['clearDrawing','normalizeDrawing','returnSample'].forEach(id=>$(id).hidden=true);$('drawMode').hidden=false;
    const list=sampleList();
    if(!list.length){$('sampleMeta').textContent='Không có mẫu trong nhóm này';return;}
    sampleIndex=(sampleIndex+list.length)%list.length;
    const s=list[sampleIndex];input=Float64Array.from(binary(s.pixels),v=>v/255);
    $('samplePosition').textContent=(sampleIndex+1)+' / '+list.length;
    $('sampleMeta').textContent='Test #'+s.index+' · Nhãn thật: '+s.label;
    renderInspect();
  }
  function renderInspect(){
    const m=model(),base=forward(m,input),trace=forward(m,input,disabled),pred=argmax(trace.p),before=argmax(base.p);
    paintPixels($('inputCanvas'),input);
    paintActivations(0,trace.acts[1]);paintActivations(1,trace.acts[2]);
    $('epochLabel').textContent='epoch '+checkpoints[epochIndex].epoch+(checkpoints[epochIndex].epoch===D.best_epoch?' · chọn bằng validation':'');
    $('probabilities').innerHTML=trace.p.map((p,i)=>`<div class="prob-row ${i===pred?'winner':''}"><span>${i}</span><div class="prob-track"><div class="prob-fill" style="width:${p*100}%"></div></div><span class="prob-number">${p<.0005?'<0,1':(p*100).toFixed(1)}%</span></div>`).join('');
    $('predictedDigit').textContent=pred;$('predictionScore').textContent=pct(trace.p[pred]);
    $('selectedNeuronLabel').textContent=selectedNeuron+1;$('selectedNeuron').max=m.dims[selectedLayer+1]-1;
    $('selectedNeuron').value=selectedNeuron;$('selectedLayer').value=selectedLayer;
    $('ablateNeuron').checked=disabled[selectedLayer].has(selectedNeuron);
    const n=disabled[0].size+disabled[1].size,delta=(trace.p[before]-base.p[before])*100;
    $('interventionResult').textContent=n?`${n} neuron đang bị tắt. Dự đoán ${before} → ${pred}. Xác suất lớp ${before}: ${delta>=0?'+':''}${delta.toFixed(2)} điểm %.`:`Neuron ${selectedNeuron+1}, lớp ${selectedLayer+1}: activation = ${base.acts[selectedLayer+1][selectedNeuron].toFixed(4)}. Chưa có can thiệp.`;
    const nin=m.dims[selectedLayer],row=Array.from(m.weights[selectedLayer].slice(selectedNeuron*nin,(selectedNeuron+1)*nin),v=>v*(m.quantized?m.scales[selectedLayer][selectedNeuron]:1));
    paintPixels($('weightCanvas'),row,selectedLayer===0?28:8,'signed');
    $('weightTitle').textContent='Neuron '+(selectedNeuron+1)+' · Lớp '+(selectedLayer+1);
    $('weightNote').textContent=selectedLayer===0?'784 trọng số được xếp theo vị trí pixel. Xanh = âm; cam = dương. Thang đối xứng riêng cho neuron.':'64 trọng số từ lớp ẩn trước, xếp thành lưới 8 × 8. Lưới này không phải vị trí pixel của ảnh.';
    paintPixels($('saliencyCanvas'),saliency(m,trace,pred,disabled),28,'heat');
  }
  function setupCanvas(id){
    const c=$(id),rect=c.getBoundingClientRect(),w=Math.max(200,rect.width),h=rect.height||260,dpr=window.devicePixelRatio||1;
    c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);const ctx=c.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);return {ctx,w,h};
  }
  function plot(id,series,label,range){
    const {ctx,w,h}=setupCanvas(id),pad={l:48,r:16,t:20,b:38},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;
    const all=series.flatMap(s=>s.values),maxX=Math.max(...all.map(v=>v[0])),maxY=range?range[1]:Math.max(...all.map(v=>v[1]))*1.12,minY=range?range[0]:0;
    const xx=x=>pad.l+x/Math.max(1,maxX)*pw,yy=y=>pad.t+(maxY-y)/Math.max(1e-12,maxY-minY)*ph;
    ctx.font='11px system-ui';ctx.textBaseline='middle';ctx.fillStyle='#64727d';ctx.textAlign='right';
    for(let i=0;i<=4;i++){const y=minY+(maxY-minY)*i/4,py=yy(y);ctx.strokeStyle='#edf0f4';ctx.beginPath();ctx.moveTo(pad.l,py);ctx.lineTo(w-pad.r,py);ctx.stroke();ctx.fillText(maxY>10?y.toFixed(0):y.toFixed(maxY<1?2:1),pad.l-8,py);}
    ctx.textAlign='center';const step=w<400?6:3;for(let e=0;e<=maxX;e+=step)ctx.fillText(e,xx(e),h-22);
    ctx.fillText('Epoch',pad.l+pw/2,h-7);ctx.textAlign='left';ctx.fillText(label,pad.l,7);
    for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();s.values.forEach((v,i)=>{if(i)ctx.lineTo(xx(v[0]),yy(v[1]));else ctx.moveTo(xx(v[0]),yy(v[1]));});ctx.stroke();const last=s.values[s.values.length-1];ctx.beginPath();ctx.arc(xx(last[0]),yy(last[1]),3,0,2*Math.PI);ctx.fillStyle=s.color;ctx.fill();}
  }
  function renderLearning(){
    const hist=D.history;
    plot('accuracyChart',[{values:hist.map(r=>[r.epoch,r.train_accuracy*100]),color:colors[0]},{values:hist.map(r=>[r.epoch,r.val_accuracy*100]),color:colors[1]}],'Accuracy (%)',[0,100]);
    plot('lossChart',[{values:hist.map(r=>[r.epoch,r.train_loss]),color:colors[0]},{values:hist.map(r=>[r.epoch,r.val_loss]),color:colors[1]}],'Cross-entropy');
    plot('zeroChart',[0,1].map(l=>({values:hist.map(r=>[r.epoch,r.layer_stats[l].zero_fraction*100]),color:colors[l]})),'Activation = 0 (%)',[0,100]);
    plot('gradientChart',[0,1,2].map(l=>({values:hist.filter(r=>r.epoch>0).map(r=>[r.epoch,r.gradient_norm[l]]),color:colors[l]})),'L2 gradient norm');
  }
  function renderBudget(){
    const budget=Number($('budgetSlider').value)*1024;$('budgetLabel').textContent=(budget/1024)+' KiB';
    const eligible=D.groups.filter(g=>g.model_bytes<=budget).sort((a,b)=>b.validation_mean-a.validation_mean),best=eligible[0];
    $('budgetRecommendation').innerHTML=best?`<span class="recommendation">${names[best.model]} · ${best.variant.toUpperCase()}</span>Ưu tiên theo accuracy validation trung bình. Tệp ${bytes(best.model_bytes)}; test ${pct(best.accuracy_mean)}.`:'Chưa có cấu hình trong thí nghiệm đáp ứng ngân sách này.';
    $('benchmarkRows').innerHTML=D.groups.map(g=>`<tr class="${g.model_bytes<=budget?'eligible':''}"><td>${names[g.model]}<br><small>${g.dims.join(' → ')}</small></td><td><span class="precision-chip ${g.variant==='int8'?'q':''}">${g.variant.toUpperCase()}</span></td><td>${(g.accuracy_mean*100).toFixed(2)} ± ${(g.accuracy_sd*100).toFixed(2)}%</td><td>${bytes(g.model_bytes)}</td><td>${g.latency_p50_ms.toFixed(2)} ms</td><td class="${g.model_bytes<=budget?'budget-yes':'budget-no'}">${g.model_bytes<=budget?'Đáp ứng':'Vượt'}</td></tr>`).join('');
    const {ctx,w,h}=setupCanvas('budgetChart'),p={l:48,r:17,t:15,b:38},pw=w-p.l-p.r,ph=h-p.t-p.b;
    const maxX=Math.ceil(Math.max(...D.groups.map(g=>g.model_bytes/1024))/50)*50,minY=Math.floor(Math.min(...D.groups.map(g=>(g.accuracy_mean-g.accuracy_sd)*100))-2),maxY=100;
    const xx=x=>p.l+x/maxX*pw,yy=y=>p.t+(maxY-y)/(maxY-minY)*ph;
    ctx.font='11px system-ui';ctx.fillStyle='#64727d';ctx.textAlign='right';ctx.textBaseline='middle';
    for(let i=0;i<=4;i++){const y=minY+(maxY-minY)*i/4;ctx.strokeStyle='#edf0f4';ctx.beginPath();ctx.moveTo(p.l,yy(y));ctx.lineTo(w-p.r,yy(y));ctx.stroke();ctx.fillText(y.toFixed(0),p.l-8,yy(y));}
    ctx.textAlign='center';for(let x=0;x<=maxX;x+=50)ctx.fillText(x,xx(x),h-22);ctx.fillText('Dung lượng tệp (KiB)',p.l+pw/2,h-6);
    const threshold=budget/1024;if(threshold<=maxX){ctx.strokeStyle='#9db6ad';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(xx(threshold),p.t);ctx.lineTo(xx(threshold),p.t+ph);ctx.stroke();ctx.setLineDash([]);}
    D.groups.forEach(g=>{const x=xx(g.model_bytes/1024),y=yy(g.accuracy_mean*100),low=yy((g.accuracy_mean-g.accuracy_sd)*100),high=yy((g.accuracy_mean+g.accuracy_sd)*100);ctx.strokeStyle=g.variant==='fp32'?colors[0]:colors[1];ctx.fillStyle=ctx.strokeStyle;ctx.globalAlpha=g.model_bytes<=budget?1:.4;ctx.lineWidth=1.4;ctx.beginPath();ctx.moveTo(x,low);ctx.lineTo(x,high);ctx.moveTo(x-3,low);ctx.lineTo(x+3,low);ctx.moveTo(x-3,high);ctx.lineTo(x+3,high);ctx.stroke();if(g.variant==='int8')ctx.fillRect(x-4,y-4,8,8);else{ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.fill();}});ctx.globalAlpha=1;
    ctx.textAlign='left';ctx.fillStyle='#64727d';ctx.fillText('Test accuracy (%)',p.l,5);
  }
  function renderEvidence(){
    const c=$('confusionChart'),ctx=c.getContext('2d');ctx.clearRect(0,0,480,430);const cell=33,ox=66,oy=25,cm=D.main.confusion,max=Math.max(...cm.flat());
    ctx.font='11px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';
    for(let i=0;i<10;i++)for(let j=0;j<10;j++){const t=cm[i][j]/max;ctx.fillStyle=`rgb(${Math.round(245-143*t)},${Math.round(242-159*t)},${Math.round(252-46*t)})`;ctx.fillRect(ox+j*cell,oy+i*cell,cell-1,cell-1);ctx.fillStyle=t>.5?'#fff':'#574e70';ctx.fillText(cm[i][j],ox+j*cell+cell/2,oy+i*cell+cell/2);}
    ctx.fillStyle='#65717a';for(let i=0;i<10;i++){ctx.fillText(i,ox-16,oy+i*cell+cell/2);ctx.fillText(i,ox+i*cell+cell/2,oy+cell*10+16);}ctx.fillText('Chữ số dự đoán',ox+165,oy+cell*10+40);ctx.save();ctx.translate(18,oy+165);ctx.rotate(-Math.PI/2);ctx.fillText('Nhãn thật',0,0);ctx.restore();
  }
  function renderCurrent(){if(currentTab==='inspect')renderInspect();if(currentTab==='learning')renderLearning();if(currentTab==='budget')renderBudget();if(currentTab==='evidence')renderEvidence();}
  function initFacts(){
    const fp=D.groups.find(g=>g.model==='observatory_64_32'&&g.variant==='fp32'),q=D.groups.find(g=>g.model==='observatory_64_32'&&g.variant==='int8');
    $('heroAccuracy').textContent=pct(fp.accuracy_mean);$('heroCompression').textContent=(fp.model_bytes/q.model_bytes).toFixed(2)+'×';
    $('bestEpochText').textContent='Epoch '+D.best_epoch+' / 18';$('parameterText').textContent=D.main.parameters.toLocaleString('en-US');$('trainingTimeText').textContent=D.main.training_seconds.toFixed(1)+' giây';
    $('compressionFacts').innerHTML=[['Tệp FP32',bytes(fp.model_bytes)],['Tệp INT8',bytes(q.model_bytes)],['Thay đổi test accuracy',(100*(q.accuracy_mean-fp.accuracy_mean)).toFixed(3)+' điểm %'],['Tỉ lệ dung lượng',(fp.model_bytes/q.model_bytes).toFixed(2)+'×']].map(([label,v])=>`<div class="fact"><span>${label}</span><strong>${v}</strong></div>`).join('');
    $('robustnessFacts').innerHTML=[['Ảnh test gốc',pct(D.robustness.clean.accuracy)],['Dịch sang phải 2 pixel',pct(D.robustness.shifted.accuracy)]].map(([label,v])=>`<div class="fact"><span>${label}</span><strong>${v}</strong></div>`).join('');
    const top=[...D.ablation.items].sort((a,b)=>b.accuracy_drop_pp-a.accuracy_drop_pp).slice(0,5),max=Math.max(...top.map(r=>r.accuracy_drop_pp),.01);
    $('ablationRanking').innerHTML=top.map(r=>`<div class="rank-row"><span>L${r.layer+1} / N${r.neuron+1}</span><div class="rank-track"><span style="width:${100*r.accuracy_drop_pp/max}%"></span></div><span>−${r.accuracy_drop_pp.toFixed(2)} đ.%</span></div>`).join('');
    $('environmentFacts').textContent=`Môi trường đã đo: ${D.protocol.cpu}. Python ${D.protocol.python}; NumPy ${D.protocol.numpy}; BLAS 1 thread. Seed 42, 43, 44. Nguồn gốc: commit ${D.protocol.source_commit.slice(0,12)}. Các phép đo thời gian phụ thuộc môi trường này.`;
  }
  document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>{
    currentTab=b.dataset.tab;document.querySelectorAll('[data-tab]').forEach(t=>{t.classList.toggle('active',t===b);t.setAttribute('aria-selected',t===b?'true':'false');});
    document.querySelectorAll('.tab-panel').forEach(p=>p.hidden=p.id!==currentTab);requestAnimationFrame(renderCurrent);
  }));
  $('sampleFilter').addEventListener('change',e=>{filter=e.target.value;sampleIndex=0;selectSample();});
  $('prevSample').addEventListener('click',()=>{sampleIndex--;selectSample();});$('nextSample').addEventListener('click',()=>{sampleIndex++;selectSample();});
  $('epochSlider').max=checkpoints.length-1;$('epochSlider').value=epochIndex;
  $('epochSlider').addEventListener('input',e=>{epochIndex=Number(e.target.value);renderInspect();});
  $('precision').addEventListener('change',()=>{const isQ=$('precision').value==='int8';$('epochSlider').disabled=isQ;if(isQ){epochIndex=checkpoints.findIndex(m=>m.epoch===D.best_epoch);$('epochSlider').value=epochIndex;}renderInspect();});
  $('selectedLayer').addEventListener('change',e=>{selectedLayer=Number(e.target.value);selectedNeuron=Math.min(selectedNeuron,model().dims[selectedLayer+1]-1);renderInspect();});
  $('selectedNeuron').addEventListener('input',e=>{selectedNeuron=Number(e.target.value);renderInspect();});
  $('ablateNeuron').addEventListener('change',e=>{if(e.target.checked)disabled[selectedLayer].add(selectedNeuron);else disabled[selectedLayer].delete(selectedNeuron);renderInspect();});
  $('resetInterventions').addEventListener('click',()=>{disabled=[new Set(),new Set()];renderInspect();});
  [0,1].forEach(l=>$('layer'+l+'Canvas').addEventListener('click',e=>{const rect=e.target.getBoundingClientRect(),x=(e.clientX-rect.left)*240/rect.width,y=(e.clientY-rect.top)*240/rect.height,rows=model().dims[l+1]/8,oy=(240-rows*30)/2,col=Math.floor(x/30),row=Math.floor((y-oy)/30);if(col>=0&&col<8&&row>=0&&row<rows){selectedLayer=l;selectedNeuron=row*8+col;renderInspect();}}));
  $('drawMode').addEventListener('click',()=>{drawing=true;input=new Float64Array(784);document.body.classList.add('draw-active');$('sampleMeta').textContent='Nét vẽ mới · chưa có nhãn thật';['clearDrawing','normalizeDrawing','returnSample'].forEach(id=>$(id).hidden=false);$('drawMode').hidden=true;renderInspect();});
  $('clearDrawing').addEventListener('click',()=>{input.fill(0);renderInspect();});$('returnSample').addEventListener('click',selectSample);
  function brush(e){if(!drawing)return;const r=$('inputCanvas').getBoundingClientRect(),x=(e.clientX-r.left)*28/r.width,y=(e.clientY-r.top)*28/r.height;
    for(let j=Math.max(0,Math.floor(y-2));j<=Math.min(27,Math.ceil(y+2));j++)for(let i=Math.max(0,Math.floor(x-2));i<=Math.min(27,Math.ceil(x+2));i++){const d=Math.hypot(i+.5-x,j+.5-y);input[j*28+i]=Math.max(input[j*28+i],Math.max(0,Math.min(1,1.7-d)));}renderInspect();}
  $('inputCanvas').addEventListener('pointerdown',e=>{if(!drawing)return;isDrawing=true;e.target.setPointerCapture(e.pointerId);brush(e);});
  $('inputCanvas').addEventListener('pointermove',e=>{if(isDrawing)brush(e);});['pointerup','pointercancel'].forEach(type=>$('inputCanvas').addEventListener(type,()=>{isDrawing=false;}));
  $('normalizeDrawing').addEventListener('click',()=>{const indices=Array.from(input,(_,i)=>i).filter(i=>input[i]>.1);if(!indices.length)return;const xs=indices.map(i=>i%28),ys=indices.map(i=>Math.floor(i/28)),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),scale=20/Math.max(x1-x0+1,y1-y0+1),nw=(x1-x0+1)*scale,nh=(y1-y0+1)*scale,next=new Float64Array(784);for(let y=0;y<28;y++)for(let x=0;x<28;x++){const sx=x0+(x-(28-nw)/2)/scale,sy=y0+(y-(28-nh)/2)/scale,ix=Math.floor(sx),iy=Math.floor(sy);if(ix>=0&&ix<28&&iy>=0&&iy<28)next[y*28+x]=input[iy*28+ix];}input=next;renderInspect();});
  $('budgetSlider').addEventListener('input',renderBudget);
  $('downloadCSV').addEventListener('click',()=>{const fields=['model','variant','parameters','model_bytes','weight_bytes','accuracy_mean','accuracy_sd','validation_mean','latency_p50_ms','latency_p95_ms'],csv=[fields.join(','),...D.groups.map(g=>fields.map(f=>g[f]).join(','))].join('\n'),a=document.createElement('a'),url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.href=url;a.download='neural_observatory_aggregate.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(renderCurrent,100);});
  initFacts();selectSample();
  // Explicit test interface; all functions operate on public experiment data.
  window.Observatory={forward,saliency,decode,checkpoints,quantized,data:D,getState:()=>({epochIndex,selectedLayer,selectedNeuron,input:Array.from(input),disabled:disabled.map(s=>[...s])})};
})();
