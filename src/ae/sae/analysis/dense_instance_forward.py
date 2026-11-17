"""Source-only bounded expansion of DenseNet's feature-list forward semantics.

No target model imports or execution occur in this module. Source drift fails closed.
"""
from __future__ import annotations

import ast
import hashlib
from collections import defaultdict

from .instance_forward import _Expander, _Value, _check, _method, _self_attr, StaticPathUnsupported
from ..linking.project import Unit, span

# Exact external implementation that was independently audited at this stage.
_PINNED_SOURCE_SHA256 = "76a71797b7873088c27ca0458588861c197528e86421f57525dc8e3ad771c59a"


def _call_site(parent: ast.AST, member: str) -> ast.Call:
    matches = [n for n in ast.walk(parent) if isinstance(n, ast.Call) and
               (_self_attr(n.func) == member if member.startswith(('norm','relu','conv'))
                else isinstance(n.func,ast.Attribute) and n.func.attr == member)]
    _check(len(matches)==1, f'No unique source operation {member}')
    return matches[0]


def _named_class(root: ast.Module, name: str) -> ast.ClassDef:
    matches=[x for x in root.body if isinstance(x,ast.ClassDef) and x.name==name]
    _check(len(matches)==1,f'Cannot resolve class {name}')
    return matches[0]


def _source_sites(unit: Unit):
    root=unit.tree
    model=_named_class(root,'DenseNet')
    layer=_named_class(root,'_DenseLayer')
    block=_named_class(root,'_DenseBlock')
    transition=_named_class(root,'_Transition')
    init=_method(model,'__init__')
    model_forward=_method(model,'forward')
    bn=_method(layer,'bn_function')
    forward_overloads=[n for n in layer.body if isinstance(n,ast.FunctionDef) and n.name=='forward']
    _check(len(forward_overloads)==3, 'Unsupported DenseLayer overloaded forward methods')
    layer_forward=forward_overloads[-1]
    block_forward=_method(block,'forward')
    trans_init=_method(transition,'__init__')
    sites={}
    names=('conv0','norm0','relu0','pool0')
    for name in names:
        matches=[n for n in ast.walk(init) if isinstance(n,ast.Tuple) and len(n.elts)==2
                 and isinstance(n.elts[0],ast.Constant) and n.elts[0].value==name]
        _check(len(matches)==1,f'Cannot prove stem member {name}')
        sites[name]=matches[0].elts[1]
    for name in ('norm1','relu1','conv1'):
        sites['dense_'+name]=_call_site(bn,name)
    for name in ('norm2','relu2','conv2'):
        sites['dense_'+name]=_call_site(layer_forward,name)
    for name in ('norm','relu','conv','pool'):
        matches=[n for n in trans_init.body if isinstance(n,ast.Assign) and
                 len(n.targets)==1 and _self_attr(n.targets[0])==name]
        _check(len(matches)==1,f'Cannot prove transition {name}')
        sites['transition_'+name]=matches[0].value
    add_norm=[n for n in ast.walk(init) if isinstance(n,ast.Call) and len(n.args)>=1
              and isinstance(n.args[0],ast.Constant) and n.args[0].value=='norm5'
              and isinstance(n.func,ast.Attribute) and n.func.attr=='add_module']
    _check(len(add_norm)==1,'Cannot prove final norm5')
    sites['norm5']=add_norm[0]
    sites['dense_cat']=_call_site(bn,'cat')
    sites['block_cat']=_call_site(block_forward,'cat')
    for name in ('relu','adaptive_avg_pool2d','flatten','classifier'):
        matches=[n for n in ast.walk(model_forward) if isinstance(n,ast.Call) and
                 ((name=='classifier' and _self_attr(n.func)=='classifier') or
                  (name!='classifier' and isinstance(n.func,ast.Attribute) and
                   n.func.attr==name and isinstance(n.func.value,ast.Name) and
                   n.func.value.id==('torch' if name=='flatten' else 'F')))]
        _check(len(matches)==1,f'Cannot prove output operation {name}')
        sites['head_'+name]=matches[0]
    sites['input']=model_forward
    returns=[n for n in model_forward.body if isinstance(n,ast.Return)]
    _check(len(returns)==1,'Cannot prove DenseNet output')
    sites['output']=returns[0]
    return sites


def _factory_parameters(unit: Unit, info: dict) -> tuple[int,tuple[int,...],int,int,int]:
    """Cross-check constructor blueprint against the selected source factory."""
    _check(info.get('resolution')=='conditional' and info.get('block_type')=='_DenseBlock'
           and info.get('member_type')=='_DenseLayer', 'No supported DenseNet container factory')
    factories=[n for n in unit.tree.body if isinstance(n,ast.FunctionDef) and n.name=='densenet121']
    _check(len(factories)==1, 'DenseNet121 factory unresolved')
    exprs=[n for n in ast.walk(factories[0]) if isinstance(n,ast.Call) and
           isinstance(n.func,ast.Name) and n.func.id=='_densenet']
    _check(len(exprs)==1,'Missing unique _densenet call')
    args=exprs[0].args
    _check(len(args)==5,'Unexpected factory arguments')
    growth=ast.literal_eval(args[0]);counts=tuple(ast.literal_eval(args[1]));initial=ast.literal_eval(args[2])
    _check((growth,counts,initial)==(32,(6,12,24,16),64), 'Unsupported default factory configuration')
    _check(tuple(info.get('stage_counts',[]))==counts, 'Dense block construction disagrees with factory')
    model=_named_class(unit.tree,'DenseNet')
    init=_method(model,'__init__')
    defaults=dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],init.args.defaults))
    _check(ast.literal_eval(defaults['bn_size'])==4 and ast.literal_eval(defaults['drop_rate'])==0
           and ast.literal_eval(defaults['memory_efficient']) is False
           and ast.literal_eval(defaults['num_classes'])==1000,
           'Unsupported bottleneck/dropout/checkpoint/class-count defaults')
    _check(len(info.get('stages',[]))==len(counts) and all(len(s['member_instance_ids'])==count
           for s,count in zip(info['stages'],counts)), 'Incomplete static member instances')
    return growth,counts,initial,4,1000


class _DenseExpander(_Expander):
    def __init__(self, unit:Unit, info:dict, input_shape:tuple[int,...], sites:dict, ir_nodes:list[dict]):
        self.unit=unit
        self.info=info
        self.input_shape=input_shape
        self.sites=sites
        self.nodes=[];self.edges=[];self.counts={}
        self.source_site_index={}
        for n in ir_nodes:
            for ref in n.get('source_refs',[]):
                if ref.get('path')==unit.path and ref.get('precision')=='exact':
                    k=(ref['start_line'],ref['start_column'],ref['end_line'],ref['end_column'])
                    self.source_site_index.setdefault(k,[]).append(n['id'])

    def call(self,name:str,value:_Value,out_channels:int|None=None, *,kernel:int=1,stride:int=1,padding:int=0,
             module_id:str|None=None,site:str|None=None) -> _Value:
        shape=(self._conv(value.shape,out_channels,kernel,stride,padding)
               if out_channels is not None else value.shape)
        return self.emit('call_module',name,[value],shape,self.sites[site or name],module_instance=name)

    def cat(self,values:list[_Value],site:str)->_Value:
        _check(bool(values) and all(len(x.shape)==4 for x in values),'Unknown dense concat inputs')
        n,_,h,w=values[0].shape
        _check(all((x.shape[0],x.shape[2],x.shape[3])==(n,h,w) for x in values),
               'Dense concat axis mismatch')
        return self.emit('call_function','cat',values,(n,sum(x.shape[1] for x in values),h,w),self.sites[site])

    def run(self) -> dict:
        _,counts,c,_,classes=_factory_parameters(self.unit,self.info)
        _check(len(self.input_shape)==4 and all(type(x)==int and x>0 for x in self.input_shape)
               and self.input_shape[1]==3,'Expected concrete NCHW RGB tensor')
        a=self.emit('placeholder','x',[],self.input_shape,self.sites['input'])
        a=self.call('features.conv0',a,64,kernel=7,stride=2,padding=3,site='conv0')
        a=self.call('features.norm0',a,site='norm0')
        a=self.call('features.relu0',a,site='relu0')
        a=self.call('features.pool0',a,a.shape[1],kernel=3,stride=2,padding=1,site='pool0')
        for stage_idx,count in enumerate(counts,1):
            features=[a]
            stage_info=self.info['stages'][stage_idx-1]
            for i in range(1,count+1):
                prefix=f'features.denseblock{stage_idx}.denselayer{i}'
                cat=self.cat(features,'dense_cat')
                p=self.call(prefix+'.norm1',cat,site='dense_norm1')
                p=self.call(prefix+'.relu1',p,site='dense_relu1')
                p=self.call(prefix+'.conv1',p,128,site='dense_conv1')
                p=self.call(prefix+'.norm2',p,site='dense_norm2')
                p=self.call(prefix+'.relu2',p,site='dense_relu2')
                p=self.call(prefix+'.conv2',p,32,kernel=3,padding=1,site='dense_conv2')
                features.append(p)
                instance_id=stage_info['member_instance_ids'][i-1]
                for node in self.nodes[-7:]:
                    node['factory_instance_id']=instance_id
            a=self.cat(features,'block_cat')
            if stage_idx<4:
                prefix=f'features.transition{stage_idx}'
                a=self.call(prefix+'.norm',a,site='transition_norm')
                a=self.call(prefix+'.relu',a,site='transition_relu')
                a=self.call(prefix+'.conv',a,a.shape[1]//2,site='transition_conv')
                shape=a.shape
                _check(shape[2]>=2 and shape[3]>=2,'Pool would yield zero spatial axis')
                a=self.call(prefix+'.pool',a,shape[1],kernel=2,stride=2,site='transition_pool')
        a=self.call('features.norm5',a,site='norm5')
        a=self.emit('call_function','relu',[a],a.shape,self.sites['head_relu'])
        a=self.emit('call_function','adaptive_avg_pool2d',[a],(*a.shape[:2],1,1),self.sites['head_adaptive_avg_pool2d'])
        _check(a.shape[1]==1024,'Unknown final DenseNet classifier width')
        a=self.emit('call_function','flatten',[a],(a.shape[0],a.shape[1]),self.sites['head_flatten'])
        a=self.emit('call_module','classifier',[a],(a.shape[0],classes),
                    self.sites['head_classifier'],module_instance='classifier')
        a=self.emit('output','output',[a],a.shape,self.sites['output'])
        return {'schema_version':'1.0','status':'expanded_static',
                'scope':'pinned_source_proven_default_densenet121_factory_path',
                'entrypoint_factory':self.info['factory_definition_id'],
                'source_path':self.unit.path,'source_ref':span(self.unit,self.sites['input']),
                'input_conditions':{'input_shape':list(self.input_shape),'default_factory_arguments':True},
                'nodes':self.nodes,'edges':self.edges,'unresolved':[],
                'output_shape':list(a.shape)}


def expand_dense_forward(units:list[Unit], info:dict|None, input_shape:tuple[int,...]|None,
                         ir_nodes:list[dict]) -> dict|None:
    if input_shape is None or info is None:
        return None
    found=[u for u in units if any(isinstance(n,ast.ClassDef) and n.name=='DenseNet' for n in u.tree.body)]
    if len(found)!=1:
        return {'schema_version':'1.0','status':'unsupported','issues':['DENSENET_SOURCE_UNRESOLVED'],
                'nodes':[],'edges':[]}
    unit=found[0]
    try:
        _check(hashlib.sha256(unit.source.raw).hexdigest()==_PINNED_SOURCE_SHA256,
               'Source differs from audited TorchVision 0.25.0 DenseNet implementation')
        sites=_source_sites(unit)
        _factory_parameters(unit,info)
        return _DenseExpander(unit,info,input_shape,sites,ir_nodes).run()
    except (StaticPathUnsupported,KeyError,IndexError,TypeError,ValueError,StopIteration) as exc:
        return {'schema_version':'1.0','status':'unsupported',
                'issues':[f'{type(exc).__name__}: {exc}'],'nodes':[],'edges':[]}
