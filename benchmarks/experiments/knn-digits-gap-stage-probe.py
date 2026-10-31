import hashlib,json,statistics,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_info
ROOT=Path('/Users/tunguz/Programming/mlxlearn');sys.path.insert(0,str(ROOT))
from benchmarks.knn_broad_fixtures import workload_specs,_real
from mlxlearn._kernels import neighbors_cpu,_neighbors_cpu_native
from mlxlearn.neighbors import KNeighborsRegressor
from mlxlearn.neighbors._base import compute_weights

def measure(fn,n=151):
 for _ in range(15):fn()
 vals=[]
 for _ in range(n):
  t=time.perf_counter_ns();fn();vals.append((time.perf_counter_ns()-t)/1000)
 return {'median_us':statistics.median(vals),'samples_us':vals}
rows=[]
for s in workload_specs():
 if s['family']!='real_digits_raw_k5':continue
 a=_real(s)[0];X,Q,y=a['X'],a['Q'],a['targets']
 index=neighbors_cpu.CPUIndex(X);index.query(Q,5)
 model=KNeighborsRegressor(n_neighbors=5,algorithm='brute',weights='distance').fit(X,y)
 d,i=model.kneighbors(Q); expected=cdist(Q.astype('d'),X.astype('d'),'sqeuclidean')
 ids=np.argsort(expected,axis=1,kind='stable')[:,:5]
 np.testing.assert_array_equal(i,ids)
 np.testing.assert_array_equal(d,np.sqrt(np.take_along_axis(expected,ids,axis=1)))
 dots=Q@X.T;xn=np.einsum('ij,ij->i',X,X).reshape(1,-1);qn=np.einsum('ij,ij->i',Q,Q).reshape(-1,1)
 out=np.empty((len(Q),5),np.float64);outi=np.empty((len(Q),5),np.int64)
 neighbors=model._y[i];weights=compute_weights(d,'distance'); numerator=np.sum(neighbors*weights,axis=1);denominator=weights.sum(axis=1)
 def aggregation():
  n=model._y[i];w=compute_weights(d,'distance');return np.sum(n*w,axis=1)/np.sum(w,axis=1)
 funcs={'query_certificate':lambda:index._integer_query(Q),'query_norm':lambda:np.einsum('ij,ij->i',Q,Q),
 'gemm':lambda:Q@X.T,'native_selector':lambda:_neighbors_cpu_native.select_integer_gram(dots,xn,qn,5,False,0,out,outi),
 'engine_query':lambda:index.query(Q,5),'engine_integer_gram':lambda:index._integer_gram(Q,5),
 'public_kneighbors':lambda:model.kneighbors(Q),'internal_query_indices':lambda:model._query_indices(Q,need_distance=True),
 'predict_distance':lambda:model.predict(Q),'gather':lambda:model._y[i],'weights':lambda:compute_weights(d,'distance'),
 'numerator':lambda:np.sum(neighbors*weights,axis=1),'denominator':lambda:np.sum(weights,axis=1),'division':lambda:numerator/denominator,
 'aggregation':aggregation,'engine_first_query':lambda:neighbors_cpu.CPUIndex(X).query(Q,5)}
 results={name:measure(fn) for name,fn in funcs.items()}
 row={'case':s['case']['name'],'shape':[*X.shape,len(Q)],'arrays':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in a.items()},'results':results}
 rows.append(row);print(row['case'],{k:round(v['median_us'],3) for k,v in results.items()},flush=True)
p=Path(sys.argv[1]);assert not p.exists()
result={'rows':rows,'threadpools':threadpool_info(),'native_sha256':hashlib.sha256(Path(_neighbors_cpu_native.__file__).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((ROOT/'src/mlxlearn/_kernels/_neighbors_cpu_native.cpp').read_bytes()).hexdigest(),'driver':Path(__file__).read_text(),'method':'151 retained calls after15 warmups per stage, unchanged default thread settings; component attribution only, no cross-stage additivity assumption'}
p.write_text(json.dumps(result,indent=2)+'\n')
