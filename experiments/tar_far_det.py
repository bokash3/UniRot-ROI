import numpy as np, json, os, glob
R='out/xmatch'
SRC={'pipeline_aug':'UniRot-ROI','pipeline':'Original recipe','yolo':'YOLOv8n-pose light','yolo_aug':'YOLOv8n-pose augmented','yolo_pre':'YOLOv8n-pose augmented COCO-pretrained'}
MATCH=['compcode','resnet18','miura']
FARS=[1e-3,1e-4]
fgrid=np.logspace(-4,0,200)
def load(p):
    d=np.load(p); return d['genuine'].astype(np.float64),d['impostor'].astype(np.float64)
def eer(g,i):
    g,i=np.sort(g),np.sort(i);thr=np.linspace(min(g[0],i[0]),max(g[-1],i[-1]),4000)
    far=np.searchsorted(i,thr,side='right')/len(i);frr=(len(g)-np.searchsorted(g,thr,side='left'))/len(g)
    k=int(np.argmin(np.abs(far-frr)));return 50*float(far[k]+frr[k])
def tar_at(g,i,f):
    t=np.quantile(i,f);return 100*float(np.mean(g<=t))
def det(g,i):
    t=np.quantile(i,fgrid);return [100*float(np.mean(g>x)) for x in t]  # FRR (%) at each FAR
out={};
for ds in ['vera','scut']:
    for s,name in SRC.items():
        for m in MATCH:
            for seed in (42,2025,1337):
                p=f'{R}/{ds}/{s}_seed{seed}/scores_{m}_rnseed{seed}.npz'
                if not os.path.exists(p): print('missing',p);continue
                g,i=load(p)
                out[f'{ds}|{name}|{m}|{seed}']={'eer':eer(g,i),'tar1e-3':tar_at(g,i,1e-3),'tar1e-4':tar_at(g,i,1e-4),'ng':len(g),'ni':len(i),'frr':det(g,i)}
    for m in MATCH:
        if ds!='scut':continue
        for seed in ([42,2025,1337] if m=='resnet18' else [None]):
            p=f'{R}/scut/official/scores_{m}'+(f'_rnseed{seed}' if seed else '')+'.npz'
            g,i=load(p);out[f'scut|Released SCUT-PV ROI|{m}|{seed or 0}']={'eer':eer(g,i),'tar1e-3':tar_at(g,i,1e-3),'tar1e-4':tar_at(g,i,1e-4),'ng':len(g),'ni':len(i),'frr':det(g,i)}
json.dump({'far':fgrid.tolist(),'res':out},open('out/paper_tar_far/tar_far_det.json','w'))
# summary
import collections
agg=collections.defaultdict(list)
for k,v in out.items():
    ds,n,m,sd=k.split('|');agg[(ds,n,m)].append(v)
print('ds | source | matcher | EER | TAR@0.1% | TAR@0.01%  (mean ± sd over seeds)')
for (ds,n,m),L in agg.items():
    f=lambda key:f"{np.mean([x[key] for x in L]):.1f} ± {np.std([x[key] for x in L],ddof=1) if len(L)>1 else 0:.1f}"
    print(ds,'|',n,'|',m,'|',f('eer'),'|',f('tar1e-3'),'|',f('tar1e-4'))
