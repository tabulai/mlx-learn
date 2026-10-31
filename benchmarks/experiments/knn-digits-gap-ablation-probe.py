import hashlib,importlib.util,json,statistics,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_info
ROOT=Path('/Users/tunguz/Programming/mlxlearn');sys.path.insert(0,str(ROOT))
from benchmarks.knn_broad_fixtures import workload_specs,_real
from mlxlearn._kernels import neighbors_cpu,_neighbors_cpu_native as current
from mlxlearn.neighbors import KNeighborsRegressor
spec=importlib.util.spec_from_file_location('digits_gap_baseline._neighbors_cpu_native','/tmp/mlxlearn-knn-digits-gap-baseline/_neighbors_cpu_native.cpython-313-darwin.so');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rows=[]
for s in workload_specs():
 if s['family']!='real_digits_raw_k5':continue
 a=_real(s)[0];X,Q,y=a['X'],a['Q'],a['targets'];dots=Q@X.T
 xn=np.einsum('ij,ij->i',X,X).reshape(1,-1);qn=np.einsum('ij,ij->i',Q,Q).reshape(-1,1)
 squared=np.empty((len(Q),5),np.float64);ids=np.empty((len(Q),5),np.int64)
 oracle=cdist(Q.astype('d'),X.astype('d'),'sqeuclidean');expected_ids=np.argsort(oracle,axis=1,kind='stable')[:,:5];expected_squared=np.take_along_axis(oracle,expected_ids,axis=1)
 model=KNeighborsRegressor(algorithm='brute',weights='distance').fit(X,y);index=neighbors_cpu.CPUIndex(X);index.query(Q,5)
 def select():
  neighbors_cpu._native_integer_select(dots,xn,qn,5,False,0,squared,ids)
  return squared,ids
 funcs={'selector':select,'engine':lambda:index.query(Q,5),'predict':lambda:model.predict(Q),'fresh_predict':lambda:KNeighborsRegressor(algorithm='brute',weights='distance').fit(X,y).predict(Q)}
 observations={name:{key:[] for key in funcs} for name in ['baseline','fused']}
 for rep in range(121):
  names=['baseline','fused'] if rep%2 else ['fused','baseline']
  for name in names:
   neighbors_cpu._native_integer_select=(old if name=='baseline' else current).select_integer_gram
   for key,fn in funcs.items():
    t=time.perf_counter_ns();result=fn();elapsed=(time.perf_counter_ns()-t)/1000
    if rep>=20:observations[name][key].append(elapsed)
    if key=='selector':
     np.testing.assert_array_equal(result[0],expected_squared);np.testing.assert_array_equal(result[1],expected_ids)
    elif key=='engine':np.testing.assert_array_equal(result[1],expected_ids)
 summary={name:{key:statistics.median(vals) for key,vals in operations.items()} for name,operations in observations.items()}
 rows.append({'case':s['case']['name'],'summary_us':summary,'samples_us':observations})
 print(s['case']['name'],summary,flush=True)
result={'rows':rows,'threadpools':threadpool_info(),'baseline_sha256':hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),'cpp_sha256':hashlib.sha256((ROOT/'src/mlxlearn/_kernels/_neighbors_cpu_native.cpp').read_bytes()).hexdigest(),'driver':Path(__file__).read_text(),'method':'alternating selector function bindings within one process, current Python identical,20warmup101retained pairs, each result independently audited; no process-global thread changes'}
p=Path(sys.argv[1]);assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
