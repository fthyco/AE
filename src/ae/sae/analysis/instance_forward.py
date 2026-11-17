"""Bounded source-only expansion of verified fixed factory forward paths.

Fails closed for unsupported construction, control flow, or tensor operators.
The original model is never imported or executed here.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from ..ir.canonical import stable_id
from ..linking.project import Unit, span


class StaticPathUnsupported(ValueError):
    """No complete source proof for the requested path."""


def _check(ok: bool, reason: str) -> None:
    if not ok:
        raise StaticPathUnsupported(reason)


def _self_attr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
        return node.attr
    return None


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    found = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name]
    _check(len(found) == 1, f'Method {cls.name}.{name} is not uniquely resolved')
    return found[0]


def _literal(node: ast.AST, env: dict[str, int] | None = None) -> int | tuple[int, ...]:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name) and env and node.id in env:
        return env[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        vals = tuple(_literal(x, env) for x in node.elts)
        _check(all(type(v) is int for v in vals), 'Noninteger shape parameter')
        return vals
    raise StaticPathUnsupported('Constructor parameter cannot be statically evaluated')


def _kwargs(node: ast.Call) -> dict[str, ast.AST]:
    _check(all(x.arg is not None for x in node.keywords), 'Unknown constructor **kwargs')
    return {x.arg: x.value for x in node.keywords}


def _assignment(cls: ast.ClassDef, attr: str) -> ast.AST:
    body = _method(cls, '__init__').body
    matches = [s.value for s in body if isinstance(s, ast.Assign)
               and len(s.targets) == 1 and _self_attr(s.targets[0]) == attr]
    _check(len(matches) == 1, f'{cls.name}.{attr} has no unique constructor assignment')
    return matches[0]


def _nn_call(cls: ast.ClassDef, attr: str, expected: str) -> ast.Call:
    expr = _assignment(cls, attr)
    _check(isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
           and expr.func.attr == expected, f'{cls.name}.{attr} is not {expected}')
    return expr


def _source_checks(unit: Unit, model: ast.ClassDef, block: ast.ClassDef,
                   funcs: dict[str, ast.FunctionDef], factory: dict, input_shape: tuple[int, ...]):
    """Prove construction and guarded downsample patterns before expansion."""
    _check(len(input_shape) == 4 and all(type(v) is int and v > 0 for v in input_shape),
           'Input must have four positive concrete dimensions')
    _check(input_shape[1] == 3, 'Input channel count differs from static stem')
    _check(block.name == 'BasicBlock' and model.name == 'ResNet',
           'No bounded construction proof for selected block/model implementation')
    exp = [n for n in block.body if isinstance(n, ast.AnnAssign)
           and isinstance(n.target, ast.Name) and n.target.id == 'expansion']
    _check(len(exp) == 1 and isinstance(exp[0].value, ast.Constant)
           and exp[0].value.value == 1, 'Block expansion is not 1')
    forward = _method(block, 'forward')
    _check(len(forward.body) == 10, 'Block forward control-flow template changed')
    _check(isinstance(forward.body[6], ast.If), 'Expected downsample guard missing')
    guard = forward.body[6]
    _check(isinstance(guard.test, ast.Compare) and _self_attr(guard.test.left) == 'downsample'
           and len(guard.test.ops) == 1 and isinstance(guard.test.ops[0], ast.IsNot)
           and isinstance(guard.test.comparators[0], ast.Constant)
           and guard.test.comparators[0].value is None and not guard.orelse,
           'Downsample execution guard is not proven')
    _check(len(guard.body) == 1 and isinstance(guard.body[0], ast.Assign)
           and isinstance(guard.body[0].value, ast.Call)
           and _self_attr(guard.body[0].value.func) == 'downsample',
           'Downsample call body is not proven')
    _check(isinstance(_assignment(block, 'downsample'), ast.Name)
           and _assignment(block, 'downsample').id == 'downsample',
           'Downsample is not bound directly to constructor argument')
    make_layer = _method(model, '_make_layer')
    guards = [s for s in make_layer.body if isinstance(s, ast.If)
              and isinstance(s.test, ast.BoolOp) and isinstance(s.test.op, ast.Or)
              and len(s.test.values) == 2]
    _check(len(guards) == 1, 'No unique downsample constructor guard')
    first, second = guards[0].test.values
    _check(isinstance(first, ast.Compare) and isinstance(first.left, ast.Name)
           and first.left.id == 'stride' and isinstance(first.ops[0], ast.NotEq)
           and _literal(first.comparators[0]) == 1,
           'Downsample guard does not depend on stride != 1')
    _check(isinstance(second, ast.Compare) and _self_attr(second.left) == 'inplanes'
           and isinstance(second.ops[0], ast.NotEq)
           and isinstance(second.comparators[0], ast.BinOp)
           and isinstance(second.comparators[0].op, ast.Mult)
           and isinstance(second.comparators[0].left, ast.Name)
           and second.comparators[0].left.id == 'planes',
           'Downsample guard does not check channel mismatch')
    rhs = second.comparators[0].right
    _check(isinstance(rhs, ast.Attribute) and isinstance(rhs.value, ast.Name)
           and rhs.value.id == 'block' and rhs.attr == 'expansion',
           'Downsample guard uses unsupported expansion')
    make_downsample = [s.value for s in guards[0].body if isinstance(s, ast.Assign)
                       and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                       and s.targets[0].id == 'downsample']
    _check(len(make_downsample) == 1 and isinstance(make_downsample[0], ast.Call)
           and isinstance(make_downsample[0].func, ast.Attribute)
           and make_downsample[0].func.attr == 'Sequential'
           and len(make_downsample[0].args) == 2,
           'Downsample container is not a static two-component Sequential')
    container = make_downsample[0]
    _check(isinstance(container.args[0], ast.Call)
           and isinstance(container.args[0].func, ast.Name)
           and container.args[0].func.id == 'conv1x1'
           and isinstance(container.args[1], ast.Call)
           and isinstance(container.args[1].func, ast.Name)
           and container.args[1].func.id == 'norm_layer',
           'Downsample leaves are not proven Conv1x1 and norm_layer')
    _check(len([n for n in ast.walk(make_layer) if isinstance(n, (ast.AugAssign, ast.Delete))
                 and any(_self_attr(c) == 'inplanes' for c in ast.walk(n))]) == 0,
           'Unknown channel mutations')
    # Verify all module constructor declarations and helper implementations.
    for attr, kind in [('conv1','Conv2d'),('bn1','BatchNorm2d'),('relu','ReLU'),
                       ('maxpool','MaxPool2d'),('avgpool','AdaptiveAvgPool2d'),('fc','Linear')]:
        expr = _nn_call(model, attr, kind) if attr not in ('bn1',) else _assignment(model,attr)
        if attr == 'bn1':
            _check(isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
                   and expr.func.id == 'norm_layer', 'Stem normalization is not norm_layer')
    for attr, kind in [('conv1','conv3x3'),('conv2','conv3x3')]:
        expr = _assignment(block,attr)
        _check(isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
               and expr.func.id == kind, f'{attr} is not proven conv3x3')
    for attr in ('bn1','bn2'):
        expr = _assignment(block,attr)
        _check(isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
               and expr.func.id == 'norm_layer', f'{attr} is not proven norm_layer')
    _nn_call(block, 'relu', 'ReLU')
    _check(all(name in funcs for name in ('conv1x1','conv3x3')), 'Missing convolution helpers')
    for helper, kernel in [('conv1x1',1),('conv3x3',3)]:
        fn = funcs[helper]
        returns = [s.value for s in fn.body if isinstance(s, ast.Return)]
        _check(len(returns) == 1 and isinstance(returns[0], ast.Call)
               and isinstance(returns[0].func, ast.Attribute)
               and returns[0].func.attr == 'Conv2d', f'{helper} does not return Conv2d')
        call = returns[0]
        kw = _kwargs(call)
        _check('kernel_size' in kw and _literal(kw['kernel_size']) == kernel,
               f'{helper} has unexpected kernel')
        if kernel == 3:
            _check('padding' in kw and isinstance(kw['padding'], ast.Name)
                   and kw['padding'].id == 'dilation', 'conv3x3 padding unsupported')
        else:
            _check('padding' not in kw, 'conv1x1 padding unsupported')
        _check('stride' in kw and isinstance(kw['stride'], ast.Name)
               and kw['stride'].id == 'stride', f'{helper} stride unsupported')
    # Resolve the default normalization implementation without importing the model.
    for cls in (model, block):
        initializer=_method(cls,'__init__')
        defaults=[g for g in initializer.body if isinstance(g,ast.If)
                  and isinstance(g.test,ast.Compare)
                  and isinstance(g.test.left,ast.Name)
                  and g.test.left.id=='norm_layer'
                  and len(g.test.ops)==1 and isinstance(g.test.ops[0],ast.Is)
                  and isinstance(g.test.comparators[0],ast.Constant)
                  and g.test.comparators[0].value is None]
        _check(len(defaults)==1 and len(defaults[0].body)==1
               and isinstance(defaults[0].body[0],ast.Assign)
               and len(defaults[0].body[0].targets)==1
               and isinstance(defaults[0].body[0].targets[0],ast.Name)
               and defaults[0].body[0].targets[0].id=='norm_layer'
               and isinstance(defaults[0].body[0].value,ast.Attribute)
               and defaults[0].body[0].value.attr=='BatchNorm2d',
               'Default normalization cannot be proven')
    # Prove parameters behind the finite tensor-shape transfer functions.
    stem=_nn_call(model,'conv1','Conv2d');stem_kw=_kwargs(stem)
    _check(len(stem.args)==2 and _literal(stem.args[0])==3
           and _self_attr(stem.args[1])=='inplanes'
           and all(k in stem_kw for k in ('kernel_size','stride','padding'))
           and [_literal(stem_kw[k]) for k in ('kernel_size','stride','padding')]==[7,2,3],
           'Unsupported stem convolution geometry')
    pool=_nn_call(model,'maxpool','MaxPool2d');pool_kw=_kwargs(pool)
    _check(len(pool.args)==0 and _literal(pool_kw['kernel_size'])==3
           and _literal(pool_kw['stride'])==2 and _literal(pool_kw['padding'])==1,
           'Unsupported maxpool geometry')
    avg=_nn_call(model,'avgpool','AdaptiveAvgPool2d')
    _check(len(avg.args)==1 and _literal(avg.args[0])==(1,1),
           'Unsupported average pooling geometry')
    for name, expected_args in [('conv1',['inplanes','planes','stride']),
                                ('conv2',['planes','planes'])]:
        conv=_assignment(block,name)
        _check(len(conv.args)==len(expected_args) and
               all(isinstance(a,ast.Name) and a.id==n
                   for a,n in zip(conv.args,expected_args)) and not conv.keywords,
               'Unsupported BasicBlock convolution binding')
    for name in ('bn1','bn2'):
        norm=_assignment(block,name)
        _check(len(norm.args)==1 and isinstance(norm.args[0],ast.Name)
               and norm.args[0].id=='planes', 'Unsupported block normalization channel binding')
    _check(len(container.args[0].args)==3 and
           _self_attr(container.args[0].args[0])=='inplanes'
           and isinstance(container.args[0].args[1],ast.BinOp)
           and isinstance(container.args[0].args[1].op,ast.Mult)
           and isinstance(container.args[0].args[1].left,ast.Name)
           and container.args[0].args[1].left.id=='planes'
           and isinstance(container.args[0].args[1].right,ast.Attribute)
           and container.args[0].args[1].right.attr=='expansion'
           and isinstance(container.args[0].args[2],ast.Name)
           and container.args[0].args[2].id=='stride'
           and len(container.args[1].args)==1,
           'Unsupported downsample geometry or normalization')
    # Ensure selected factory has no contradictory stage construction.
    init = _method(model,'__init__')
    initial = _assignment(model,'inplanes')
    inplanes = _literal(initial)
    _check(inplanes == 64, 'Initial stem channels cannot be proven')
    stage_specs = {}
    for stage in factory['stages']:
        expression = _assignment(model,stage['name'])
        _check(isinstance(expression, ast.Call) and _self_attr(expression.func) == '_make_layer',
               'Factory stage does not call _make_layer')
        _check(len(expression.args) == 3 and isinstance(expression.args[0], ast.Name)
               and expression.args[0].id == 'block', 'Unknown factory stage binding')
        planes = _literal(expression.args[1]); _check(type(planes) is int and planes > 0, 'Unknown stage channels')
        _check(isinstance(expression.args[2], ast.Subscript)
               and isinstance(expression.args[2].value, ast.Name)
               and expression.args[2].value.id == 'layers', 'Unknown stage count binding')
        kw = _kwargs(expression)
        stride = _literal(kw['stride']) if 'stride' in kw else 1
        _check(type(stride) is int and stride >= 1, 'Unknown stage stride')
        # A customized dilation path is unsupported rather than silently ignored.
        if 'dilate' in kw:
            _check(isinstance(kw['dilate'], ast.Subscript)
                   and isinstance(kw['dilate'].value, ast.Name)
                   and kw['dilate'].value.id == 'replace_stride_with_dilation',
                   'Unsupported dilation configuration')
        stage_specs[stage['name']] = {'planes':planes,'stride':stride,'start_channels':inplanes}
        inplanes=planes
    fc = _nn_call(model,'fc','Linear')
    first=fc.args[0] if fc.args else None
    _check(isinstance(first,ast.BinOp) and isinstance(first.op,ast.Mult)
           and _literal(first.left)==512 and isinstance(first.right,ast.Attribute)
           and isinstance(first.right.value,ast.Name)
           and first.right.value.id=='block' and first.right.attr=='expansion',
           'Unsupported Linear input feature binding')
    _check(len(fc.args)==2 and isinstance(fc.args[1],ast.Name)
           and fc.args[1].id == 'num_classes', 'Unknown output classes')
    num_classes = next((a.value for a in init.args.defaults if isinstance(a,ast.Constant)
                        and a.value == 1000), None)
    _check(num_classes == 1000, 'Unknown default output class count')
    # Verify direct root forward delegates to supported implementation.
    root_forward = _method(model,'forward')
    _check(len(root_forward.body)==1 and isinstance(root_forward.body[0],ast.Return)
           and isinstance(root_forward.body[0].value,ast.Call)
           and _self_attr(root_forward.body[0].value.func) == '_forward_impl',
           'Root forward has unsupported delegation')
    return stage_specs, forward, _method(model,'_forward_impl'), container


@dataclass
class _Value:
    ident: str
    shape: tuple[int,...]


class _Expander:
    def __init__(self, unit: Unit, factory: dict, stages: dict,
                 block_forward: ast.FunctionDef, root_forward: ast.FunctionDef,
                 input_shape: tuple[int,...], ir_nodes: list[dict]):
        self.unit=unit;self.factory=factory;self.stages=stages
        self.block_forward=block_forward;self.root_forward=root_forward
        self.input_shape=input_shape
        self.nodes=[];self.edges=[];self.counts={}
        self.source_site_index={}
        for n in ir_nodes:
            for ref in n.get('source_refs',[]):
                if ref.get('path')==unit.path and ref.get('precision')=='exact':
                    k=(ref['start_line'],ref['start_column'],ref['end_line'],ref['end_column'])
                    self.source_site_index.setdefault(k,[]).append(n['id'])

    def emit(self, op: str, target: str, inputs: list[_Value], shape: tuple[int,...],
             site: ast.AST, *, module_instance: str | None = None) -> _Value:
        ref=span(self.unit,site)
        key=(target,ref['start_byte'],ref['end_byte'])
        count=self.counts.get(key,0)
        self.counts[key]=count+1
        ident=stable_id('instance-forward/v1',self.unit.path,op,target,ref['start_byte'],count)
        lookup=(ref['start_line'],ref['start_column'],ref['end_line'],ref['end_column'])
        self.nodes.append({'id':ident,'op':op,'target':target,'call_ordinal':count,
                           'module_instance':module_instance, 'source_refs':[ref],
                           'source_ir_ids':sorted(set(self.source_site_index.get(lookup,[]))),
                           'shape':list(shape),'input_ids':[x.ident for x in inputs]})
        for ordinal,inp in enumerate(inputs):
            edge_id=stable_id('instance-forward-edge/v1',inp.ident,ident,ordinal)
            self.edges.append({'id':edge_id,'source':inp.ident,'target':ident,'input_slot':ordinal})
        return _Value(ident,shape)

    @staticmethod
    def _conv(shape:tuple[int,...],channels:int,kernel:int,stride:int,padding:int) -> tuple[int,...]:
        _check(len(shape)==4,'Conv input is not rank-4')
        b, _, h, w=shape
        oh=(h+2*padding-(kernel-1)-1)//stride+1
        ow=(w+2*padding-(kernel-1)-1)//stride+1
        _check(oh>0 and ow>0, 'Convolution output spatial dimension not positive')
        return b,channels,oh,ow

    def _call(self, scope: str, member: str, value: _Value, site: ast.AST,
              *, stage:dict|None=None, block_pos:int=0) -> _Value:
        path=f'{scope}.{member}' if scope else member
        if scope and scope.startswith('layer'):
            _check(stage is not None,'Missing static block configuration')
            out_channels=stage['planes']
            stride=stage['stride'] if block_pos==0 else 1
            if member=='conv1': shape=self._conv(value.shape,out_channels,3,stride,1)
            elif member=='conv2':shape=self._conv(value.shape,out_channels,3,1,1)
            elif member in {'bn1','bn2','relu'}:shape=value.shape
            else:raise StaticPathUnsupported('Unknown BasicBlock member '+member)
        else:
            if member=='conv1':shape=self._conv(value.shape,64,7,2,3)
            elif member=='maxpool':shape=self._conv(value.shape,value.shape[1],3,2,1)
            elif member=='avgpool':shape=(*value.shape[:2],1,1)
            elif member=='fc':
                _check(len(value.shape)==2 and value.shape[1]==512,'Unknown Linear input dimension')
                shape=(value.shape[0],1000)
            elif member in {'bn1','relu'}:shape=value.shape
            else:raise StaticPathUnsupported('Unknown root member '+member)
        return self.emit('call_module',path,[value],shape,site,module_instance=path)

    def _block(self, path:str, value:_Value, stage:dict, pos:int) -> _Value:
        env={'x':value}
        # First block has a projection precisely when _make_layer's proven condition is true.
        downsample=(pos==0 and (stage['stride']!=1 or stage['start_channels']!=stage['planes']))
        for stmt in self.block_forward.body:
            if isinstance(stmt,ast.Assign):
                _check(len(stmt.targets)==1 and isinstance(stmt.targets[0],ast.Name),
                       'Unsupported block assignment')
                dest=stmt.targets[0].id
                if isinstance(stmt.value,ast.Name):
                    _check(stmt.value.id in env, 'Unknown block value')
                    env[dest]=env[stmt.value.id]
                elif isinstance(stmt.value,ast.Call):
                    call=stmt.value;attr=_self_attr(call.func)
                    _check(attr is not None and len(call.args)==1
                           and isinstance(call.args[0],ast.Name) and call.args[0].id in env,
                           'Unsupported block call')
                    env[dest]=self._call(path,attr,env[call.args[0].id],call,stage=stage,block_pos=pos)
                else:raise StaticPathUnsupported('Unknown block assignment expression')
            elif isinstance(stmt,ast.If):
                _check(stmt is self.block_forward.body[6], 'Unsupported block condition')
                if downsample:
                    _check(len(stmt.body)==1 and isinstance(stmt.body[0],ast.Assign),
                           'Unknown projection guard body')
                    call=stmt.body[0].value
                    _check(isinstance(call,ast.Call) and len(call.args)==1
                           and isinstance(call.args[0],ast.Name) and call.args[0].id=='x',
                           'Unknown projection input')
                    begin=env['x'];new_shape=self._conv(begin.shape,stage['planes'],1,stage['stride'],0)
                    branch=self.emit('call_module',path+'.downsample.0',[begin],new_shape,call,
                                     module_instance=path+'.downsample.0')
                    branch=self.emit('call_module',path+'.downsample.1',[branch],new_shape,call,
                                     module_instance=path+'.downsample.1')
                    env['identity']=branch
            elif isinstance(stmt,ast.AugAssign):
                _check(isinstance(stmt.op,ast.Add) and isinstance(stmt.target,ast.Name)
                       and isinstance(stmt.value,ast.Name), 'Unknown augmented operation')
                _check(stmt.target.id in env and stmt.value.id in env,'Unknown Add operand')
                a,b=env[stmt.target.id],env[stmt.value.id]
                _check(a.shape==b.shape,'Residual Add operand shape mismatch')
                env[stmt.target.id]=self.emit('call_function','add',[a,b],a.shape,stmt)
            elif isinstance(stmt,ast.Return):
                _check(isinstance(stmt.value,ast.Name) and stmt.value.id in env,
                       'Unknown block return')
                return env[stmt.value.id]
            else:raise StaticPathUnsupported('Unknown block forward statement')
        raise StaticPathUnsupported('Missing block return')

    def _stage(self,name:str,value:_Value) -> _Value:
        stage=self.stages[name]
        info=next(s for s in self.factory['stages'] if s['name']==name)
        for idx,ident in enumerate(info['block_instance_ids']):
            path=f'{name}.{idx}'
            value=self._block(path,value,stage,idx)
            # Each expanded operator is linked to its proven factory instance ID.
            for node in self.nodes:
                if node['target'].startswith(path+'.') and 'factory_instance_id' not in node:
                    node['factory_instance_id']=ident
        return value

    def run(self) -> dict:
        source_ref=span(self.unit,self.root_forward)
        inp=self.emit('placeholder','x',[],self.input_shape,self.root_forward)
        env={'x':inp}
        for stmt in self.root_forward.body:
            if isinstance(stmt,ast.Assign):
                _check(len(stmt.targets)==1 and isinstance(stmt.targets[0],ast.Name)
                       and isinstance(stmt.value,ast.Call), 'Unsupported root assignment')
                name=stmt.targets[0].id;call=stmt.value
                _check(len(call.args)>=1 and isinstance(call.args[0],ast.Name)
                       and call.args[0].id in env,'Unsupported root input binding')
                value=env[call.args[0].id]
                module=_self_attr(call.func)
                if module in self.stages:
                    _check(len(call.args)==1 and not call.keywords,'Unknown stage call signature')
                    env[name]=self._stage(module,value)
                elif module in {'conv1','bn1','relu','maxpool','avgpool','fc'}:
                    _check(len(call.args)==1 and not call.keywords,'Unknown root module call signature')
                    env[name]=self._call('',module,value,call)
                elif isinstance(call.func,ast.Attribute) and call.func.attr=='flatten':
                    _check(isinstance(call.func.value,ast.Name) and call.func.value.id=='torch'
                           and len(call.args)==2 and _literal(call.args[1])==1,
                           'Only flatten(start_dim=1) is supported')
                    _check(len(value.shape)==4,'Unexpected Flatten rank')
                    n,c,h,w=value.shape
                    env[name]=self.emit('call_function','flatten',[value],(n,c*h*w),call)
                else:raise StaticPathUnsupported('Unknown root operation')
            elif isinstance(stmt,ast.Return):
                _check(isinstance(stmt.value,ast.Name) and stmt.value.id in env,
                       'Unknown root return')
                final=self.emit('output','output',[env[stmt.value.id]],env[stmt.value.id].shape,stmt)
                return {'schema_version':'1.0','status':'expanded_static',
                        'scope':'bounded_source_proven_default_factory_path',
                        'entrypoint_factory':self.factory['factory_definition_id'],
                        'source_path':self.unit.path,
                        'source_ref':source_ref,
                        'input_conditions':{'input_shape':list(self.input_shape),'default_factory_arguments':True},
                        'nodes':self.nodes,'edges':self.edges,'unresolved':[],
                        'output_shape':list(final.shape)}
            else:raise StaticPathUnsupported('Unknown root forward statement')
        raise StaticPathUnsupported('Missing root forward return')


def expand_factory_forward(units: list[Unit], factory: dict | None,
                           input_shape: tuple[int,...] | None, ir_nodes: list[dict]):
    """Build an instance graph only if all supported source assertions hold."""
    if input_shape is None:
        return None
    if not factory or factory.get('resolution')!='conditional':
        return {'schema_version':'1.0','status':'unsupported','issues':['NO_VERIFIED_FACTORY'],
                'nodes':[],'edges':[]}
    sources=[]
    for unit in units:
        classes={c.name:c for c in unit.tree.body if isinstance(c,ast.ClassDef)}
        if 'ResNet' in classes and factory.get('block_type') in classes:
            sources.append((unit,classes))
    if len(sources)!=1:
        return {'schema_version':'1.0','status':'unsupported','issues':['SOURCE_CLASS_UNRESOLVED'],
                'nodes':[],'edges':[]}
    unit,classes=sources[0]
    try:
        shapes, block, root, _=_source_checks(
            unit,classes['ResNet'],classes[factory['block_type']],
            {f.name:f for f in unit.tree.body if isinstance(f,ast.FunctionDef)},
            factory,input_shape)
        return _Expander(unit,factory,shapes,block,root,input_shape,ir_nodes).run()
    except (StaticPathUnsupported,KeyError,IndexError,StopIteration) as exc:
        return {'schema_version':'1.0','status':'unsupported',
                'issues':[f'{type(exc).__name__}: {exc}'],'nodes':[],'edges':[]}
