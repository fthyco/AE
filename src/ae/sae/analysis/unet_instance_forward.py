"""Source-only bounded U-Net ResNet34 instance-forward graph.

The model is never imported or executed by this module. For now this accepts
only the audited SMP 0.6.0 / TorchVision 0.25.0 source implementations.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Any

from ..ir.canonical import stable_id
from ..linking.project import Unit, span
from .instance_forward import StaticPathUnsupported, _check, _method, _self_attr

# Digests are from the unchanged, independently executed package sources.
_PINNED = {
    'base/model.py': '40bfb1c5c606a7c5c41e51cbb391bbf8daf626c5553d407b10abacb56733fcb1',
    'base/modules.py': '2bb4ab52ba162bf96b6c07e50e0d86c9e3c7b0eea585cc444c77dd243cecea11',
    'base/heads.py': 'f39ecd51959bb93e46a8b90cd9734e961ca7cb859d943690f2ee4bc68e8574a8',
    'decoders/unet/model.py': '0ffb26ba52b6815693da81ca10f202c0a86b6d5d6ec0d5825cab450b994c8b6b',
    'decoders/unet/decoder.py': '02d75d28d371e8cf471b37826808e36fe09c4c78f4174c6a987946b4a49663c7',
    'encoders/resnet.py': 'a70cfc5a8840656aef485b557adc2eb23905cbc16ebafe8e16d967fb6fcbadc0',
    'torchvision/resnet.py': '9faf2f321fdaf7edbd791aedc6e3610c7edf9c892112cbd75861beae1e9518d1',
}


@dataclass
class _Data:
    ident: str
    shape: tuple[int, ...] | None
    python_value: Any = None


class _Source:
    def __init__(self, units: list[Unit], ir_nodes: list[dict]):
        self.units = {}
        for suffix in _PINNED:
            matches = [u for u in units if u.path == suffix or u.path.endswith('/' + suffix)]
            _check(len(matches) == 1, 'Source file not uniquely resolved: ' + suffix)
            unit = matches[0]
            _check(hashlib.sha256(unit.source.raw).hexdigest() == _PINNED[suffix],
                   'Audited upstream source changed: ' + suffix)
            self.units[suffix] = unit
        self.ir_site_ids = {}
        for n in ir_nodes:
            for ref in n.get('source_refs', []):
                if ref.get('precision') != 'exact':
                    continue
                key = (ref['path'], ref['start_line'], ref['start_column'],
                       ref['end_line'], ref['end_column'])
                self.ir_site_ids.setdefault(key, set()).add(n['id'])

    def unit(self, suffix: str) -> Unit:
        return self.units[suffix]

    def class_(self, suffix: str, name: str) -> ast.ClassDef:
        classes = [x for x in self.unit(suffix).tree.body
                   if isinstance(x, ast.ClassDef) and x.name == name]
        _check(len(classes) == 1, 'Unknown class: ' + name)
        return classes[0]

    def method(self, suffix: str, cls: str, method: str) -> tuple[Unit, ast.FunctionDef]:
        return self.unit(suffix), _method(self.class_(suffix, cls), method)

    @staticmethod
    def site(candidates: list[ast.AST], what: str, ordinal: int = 0) -> ast.AST:
        _check(len(candidates) > ordinal, 'Source operation missing: ' + what)
        return candidates[ordinal]

    def calls(self, suffix: str, cls: str, method: str,
              *, member: str | None = None, func: str | None = None) -> list[tuple[Unit, ast.Call]]:
        unit, f = self.method(suffix, cls, method)
        found = []
        for node in ast.walk(f):
            if not isinstance(node, ast.Call):
                continue
            if member is not None and _self_attr(node.func) != member:
                continue
            if func is not None and not ((isinstance(node.func, ast.Attribute)
                                           and node.func.attr == func) or
                                          (isinstance(node.func, ast.Name) and node.func.id==func)):
                continue
            found.append((unit, node))
        return sorted(found, key=lambda c: (c[1].lineno, c[1].col_offset))

    def call(self, suffix: str, cls: str, method: str, *, member: str | None = None,
             func: str | None = None, ordinal: int = 0) -> tuple[Unit, ast.Call]:
        choices = self.calls(suffix, cls, method, member=member, func=func)
        _check(len(choices) > ordinal, 'No uniquely supported call: '+str((suffix, cls, method, member, func)))
        return choices[ordinal]


class _UnetExpander:
    def __init__(self, sources: _Source, input_shape: tuple[int, ...]):
        self.src = sources
        self.input_shape = input_shape
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.counts: dict[tuple, int] = {}
        self.site_map: dict[str, tuple[Unit, ast.AST]] = {}
        self._prepare_sites()

    def _prepare_sites(self):
        src = self.src
        self.site_map['input'] = src.method('base/model.py', 'SegmentationModel', 'forward')
        model_unit, forward = src.method('base/model.py', 'SegmentationModel', 'forward')
        returns = [n for n in forward.body if isinstance(n, ast.Return)]
        # Branch-specific returns: the last one is the no-aux segmentation return.
        _check(len(returns)==1 and isinstance(returns[0].value,ast.Name)
               and returns[0].value.id=='masks', 'Model output branch is not proven')
        self.site_map['output'] = model_unit, returns[0]
        # Constructor source proof by audited bytes plus explicit default topology checks.
        unet = src.class_('decoders/unet/model.py', 'Unet')
        init = _method(unet, '__init__')
        args = init.args.args
        defaults = dict(zip([a.arg for a in args][-len(init.args.defaults):], init.args.defaults))
        expected = {'encoder_name':'resnet34','encoder_depth':5,
                    'decoder_use_norm':'batchnorm',
                    'decoder_channels':(256,128,64,32,16),
                    'decoder_attention_type':None,
                    'decoder_interpolation':'nearest',
                    'in_channels':3,'classes':1,'activation':None,
                    'aux_params':None}
        for name,value in expected.items():
            _check(name in defaults and ast.literal_eval(defaults[name]) == value,
                   'Unproven U-Net factory default: '+name)
        # No auxiliary classification head, no center block (resnet34 does not start with vgg).
        encoder = src.class_('encoders/resnet.py', 'ResNetEncoder')
        eforward = _method(encoder, 'forward')
        _check(len([n for n in ast.walk(eforward) if isinstance(n,ast.Return)])==1,
               'Encoder output feature list changed')
        for name in ('conv1','bn1','relu','maxpool'):
            self.site_map['encoder_'+name] = src.call('encoders/resnet.py', 'ResNetEncoder', 'forward', member=name)
        # BasicBlock forward follows the original TorchVision method.
        block_unit, block = src.method('torchvision/resnet.py', 'BasicBlock', 'forward')
        for name in ('conv1','bn1','conv2','bn2','downsample'):
            self.site_map['block_'+name] = src.call('torchvision/resnet.py','BasicBlock','forward',member=name)
        relus = src.calls('torchvision/resnet.py', 'BasicBlock','forward',member='relu')
        _check(len(relus)==2, 'Unexpected basic-block ReLU calls')
        self.site_map['block_relu_first'] = relus[0]
        self.site_map['block_relu_final'] = relus[1]
        adds=[n for n in block.body if isinstance(n,ast.AugAssign) and isinstance(n.op,ast.Add)]
        _check(len(adds)==1,'Unknown BasicBlock skip-add')
        self.site_map['block_add'] = block_unit, adds[0]
        # Decoder class and feature list reversal are separately evidenced.
        decoder_unit, decoder = src.method('decoders/unet/decoder.py','UnetDecoder','forward')
        attrs = [n for n in ast.walk(decoder) if isinstance(n,ast.Attribute) and n.attr=='shape'
                 and isinstance(n.value,ast.Name) and n.value.id=='feature']
        _check(len(attrs)==1,'Decoder feature-shape extraction changed')
        self.site_map['decoder_shape_getattr'] = decoder_unit,attrs[0]
        spatial = [n for n in ast.walk(decoder) if isinstance(n,ast.Subscript)
                   and isinstance(n.value,ast.Attribute) and n.value.attr=='shape']
        _check(len(spatial)==1,'Decoder feature-shape slice changed')
        self.site_map['decoder_shape_slice'] = decoder_unit,spatial[0]
        target = [n for n in ast.walk(decoder) if isinstance(n,ast.Subscript)
                  and isinstance(n.value,ast.Name) and n.value.id=='spatial_shapes'
                  and isinstance(n.slice, ast.BinOp) and isinstance(n.slice.op,ast.Add)]
        _check(len(target)==1,'Decoder target-shape index changed')
        self.site_map['decoder_target_index'] = decoder_unit,target[0]
        init_unit, init_decoder = src.method('decoders/unet/decoder.py','UnetDecoder','__init__')
        centers=[n for n in ast.walk(init_decoder) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Attribute) and n.func.attr=='Identity']
        _check(len(centers)==1,'Unsupported U-Net center module')
        self.site_map['decoder_center'] = init_unit,centers[0]
        for name in ('interpolate','cat'):
            self.site_map['decoder_'+name]=src.call('decoders/unet/decoder.py',
                                                   'UnetDecoderBlock','forward',func=name)
        for name in ('attention1','attention2','conv1','conv2'):
            self.site_map['decoder_'+name]=src.call('decoders/unet/decoder.py',
                                                   'UnetDecoderBlock','forward',member=name)
        # Inner container members: source backed by the original Conv2dReLU implementation.
        for name in ('Conv2d','ReLU'):
            self.site_map['inner_'+name] = src.call('base/modules.py','Conv2dReLU','__init__',func=name)
        self.site_map['inner_norm'] = src.call('base/modules.py','Conv2dReLU','__init__',func='get_norm_layer')
        self.site_map['attention_identity'] = src.call('base/modules.py','Attention','__init__',func='Identity')
        self.site_map['head_conv'] = src.call('base/heads.py','SegmentationHead','__init__',func='Conv2d')
        self.site_map['head_identity'] = src.call('base/heads.py','SegmentationHead','__init__',func='Identity')
        self.site_map['activation_identity'] = src.call('base/modules.py','Activation','__init__',func='Identity')
        # Validate that the ResNet34 dictionary defines the expected block/width sequence.
        encoder_unit = src.unit('encoders/resnet.py')
        configs = [n for n in encoder_unit.tree.body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='resnet_encoders' for t in n.targets)]
        _check(len(configs)==1 and isinstance(configs[0].value,ast.Dict),
               'resnet34 source factory table not present')
        factory=configs[0].value
        keys=[k.value for k in factory.keys if isinstance(k,ast.Constant)]
        _check('resnet34' in keys,'resnet34 encoder factory missing')
        tree=factory.values[keys.index('resnet34')]
        _check(isinstance(tree,ast.Dict),'resnet34 factory not a dictionary')
        data={k.value:v for k,v in zip(tree.keys,tree.values) if isinstance(k,ast.Constant)}
        params=data['params']
        _check(isinstance(params,ast.Dict),'No declared encoder configuration')
        fields={k.value:v for k,v in zip(params.keys,params.values) if isinstance(k,ast.Constant)}
        _check(isinstance(fields['block'],ast.Name) and fields['block'].id=='BasicBlock' and
               ast.literal_eval(fields['layers'])==[3,4,6,3] and
               ast.literal_eval(fields['out_channels'])==[3,64,64,128,256,512],
               'Encoder configuration not proven')

    def emit(self, op: str, target: str, inputs: list[_Data], shape: tuple[int,...]|None,
             site: str, *, module_instance: str|None=None, python_value: Any=None) -> _Data:
        unit,node = self.site_map[site]
        ref = span(unit,node)
        key=(unit.path, op,target,ref['start_byte'],ref['end_byte'])
        ordinal=self.counts.get(key,0)
        self.counts[key]=ordinal+1
        ident=stable_id('instance-forward/v1',unit.path,op,target,ref['start_byte'],ordinal)
        lookup=(ref['path'],ref['start_line'],ref['start_column'],ref['end_line'],ref['end_column'])
        ids=sorted(self.src.ir_site_ids.get(lookup,[]))
        if not ids:
            ids=[stable_id(unit.path,'source-file')]
        self.nodes.append({'id':ident,'op':op,'target':target,'call_ordinal':ordinal,
                           'module_instance':module_instance, 'source_refs':[ref],
                           'source_ir_ids':ids,'shape':list(shape) if shape is not None else None,
                           'value_kind':'metadata' if shape is None else 'tensor',
                           'input_ids':[v.ident for v in inputs]})
        for slot,v in enumerate(inputs):
            relation=('STATIC_INSTANCE_SHAPE_METADATA_DEPENDENCY'
                      if shape is None or v.shape is None
                      else 'STATIC_INSTANCE_TENSOR_DEPENDENCY')
            self.edges.append({'id':stable_id('instance-forward-edge/v1',v.ident,ident,slot),
                               'source':v.ident,'target':ident,'input_slot':slot,
                               'relation':relation})
        return _Data(ident,shape,python_value)

    def module(self, path: str, value: _Data, shape: tuple[int,...], site: str) -> _Data:
        return self.emit('call_module',path,[value],shape,site,module_instance=path)

    @staticmethod
    def conv_shape(shape: tuple[int,...]|None, channels: int, kernel: int,
                   stride: int=1, padding: int=0) -> tuple[int,...]:
        _check(shape is not None and len(shape)==4,'Invalid convolution input')
        n,c,h,w=shape
        oh=(h+2*padding-kernel)//stride+1
        ow=(w+2*padding-kernel)//stride+1
        _check(oh>0 and ow>0, 'Convolution output shape invalid')
        return n,channels,oh,ow

    def _encoder(self, value: _Data) -> list[_Data]:
        features=[value]
        stem=[('conv1',self.conv_shape(value.shape,64,7,stride=2,padding=3)),
              ('bn1',None),('relu',None),('maxpool',None)]
        x=value
        for member,geometry in stem:
            if member=='maxpool':
                geometry=self.conv_shape(x.shape,64,3,stride=2,padding=1)
            x=self.module('encoder.'+member,x,geometry or x.shape,'encoder_'+member)
            if member=='relu': features.append(x)
        stages=(3,4,6,3)
        for stage,count in enumerate(stages,1):
            c=64*(2**(stage-1))
            for pos in range(count):
                path=f'encoder.layer{stage}.{pos}'
                identity=x
                stride=2 if stage>1 and pos==0 else 1
                main=self.module(path+'.conv1',x,self.conv_shape(x.shape,c,3,stride,1),'block_conv1')
                main=self.module(path+'.bn1',main,main.shape,'block_bn1')
                main=self.module(path+'.relu',main,main.shape,'block_relu_first')
                main=self.module(path+'.conv2',main,self.conv_shape(main.shape,c,3,1,1),'block_conv2')
                main=self.module(path+'.bn2',main,main.shape,'block_bn2')
                if stage>1 and pos==0:
                    identity=self.module(path+'.downsample.0',identity,
                                         self.conv_shape(identity.shape,c,1,stride,0),'block_downsample')
                    identity=self.module(path+'.downsample.1',identity,identity.shape,'block_downsample')
                _check(main.shape==identity.shape,'Encoder residual tensors disagree')
                main=self.emit('call_function','add',[main,identity],main.shape,'block_add')
                x=self.module(path+'.relu',main,main.shape,'block_relu_final')
            features.append(x)
        _check([x.shape[1] for x in features]==[3,64,64,128,256,512],
               'Encoder feature-list geometry changed')
        return features

    def _decoder(self, features: list[_Data]) -> _Data:
        _check(len(features)==6,'U-Net encoder must return six features')
        spatial=[]
        for feature in features:
            attr=self.emit('call_function','getattr',[feature],None,'decoder_shape_getattr',
                           python_value=feature.shape)
            # FX getitem takes the spatial slice from feature.shape, a metadata tuple.
            spatial.append(self.emit('call_function','getitem',[attr],None,'decoder_shape_slice',
                                     python_value=tuple(feature.shape[2:])))
        reverse=features[1:][::-1]
        x=self.module('decoder.center',reverse[0],reverse[0].shape,'decoder_center')
        channels=(256,128,64,32,16)
        for pos,out_channels in enumerate(channels):
            target=spatial[-(pos+2)]
            _check(len(target.python_value)==2,'Unresolved upsample target size')
            height=self.emit('call_function','getitem',[target],None,'decoder_target_index',
                             python_value=target.python_value[0])
            width=self.emit('call_function','getitem',[target],None,'decoder_target_index',
                            python_value=target.python_value[1])
            n,_,_,_=x.shape
            height_,width_=height.python_value,width.python_value
            up_shape=(n,x.shape[1],height_,width_)
            x=self.emit('call_function','interpolate',[x,height,width],up_shape,'decoder_interpolate')
            if pos<4:
                skip=reverse[pos+1]
                _check(skip.shape[2:]==up_shape[2:], 'Decoder skip spatial mismatch')
                cat_shape=(n,x.shape[1]+skip.shape[1],height_,width_)
                x=self.emit('call_function','cat',[x,skip],cat_shape,'decoder_cat')
                x=self.module(f'decoder.blocks.{pos}.attention1.attention',x,x.shape,'attention_identity')
            for group in ('conv1','conv2'):
                for leaf in (0,1,2):
                    if leaf==0:
                        shape=self.conv_shape(x.shape,out_channels,3,padding=1)
                        site='inner_Conv2d'
                    elif leaf==1:
                        shape=x.shape;site='inner_norm'
                    else:
                        shape=x.shape;site='inner_ReLU'
                    x=self.module(f'decoder.blocks.{pos}.{group}.{leaf}',x,shape,site)
            x=self.module(f'decoder.blocks.{pos}.attention2.attention',x,x.shape,'attention_identity')
        return x

    def run(self):
        shape=self.input_shape
        _check(len(shape)==4 and all(type(x)==int and x>0 for x in shape),
               'Expected positive concrete NCHW input')
        _check(shape[1]==3 and shape[2]>=32 and shape[3]>=32 and
               shape[2]%32==0 and shape[3]%32==0,
               'Supported input requires RGB and spatial axes divisible by 32')
        x=self.emit('placeholder','x',[],shape,'input')
        features=self._encoder(x)
        x=self._decoder(features)
        x=self.module('segmentation_head.0',x,self.conv_shape(x.shape,1,3,padding=1),'head_conv')
        x=self.module('segmentation_head.1',x,x.shape,'head_identity')
        x=self.module('segmentation_head.2.activation',x,x.shape,'activation_identity')
        x=self.emit('output','output',[x],x.shape,'output')
        _check((x.shape[0],x.shape[1],x.shape[2],x.shape[3])==
               (shape[0],1,shape[2],shape[3]),'Unet output shape mismatch')
        return {'schema_version':'1.0', 'status':'expanded_static',
                'scope':'pinned_source_proven_smp_0_6_unet_resnet34_default_topology',
                'entrypoint_factory':'decoders.unet.model:Unet',
                'source_ref':span(*self.site_map['input']),
                'source_path':self.src.unit('decoders/unet/model.py').path,
                'source_paths':{k:v.path for k,v in self.src.units.items()},
                'input_conditions':{'input_shape':list(shape),
                                    'encoder_name':'resnet34','encoder_weights':'not_loaded_by_AE',
                                    'encoder_depth':5,'decoder_channels':[256,128,64,32,16],
                                    'auxiliary_head':False},
                'nodes':self.nodes,'edges':self.edges,'unresolved':[],
                'output_shape':list(x.shape)}


def expand_unet_forward(units: list[Unit], entrypoint: str | None,
                        input_shape: tuple[int, ...] | None, ir_nodes: list[dict]) -> dict | None:
    if entrypoint not in {'decoders.unet.model:Unet','segmentation_models_pytorch.decoders.unet.model:Unet'}:
        return None
    if input_shape is None:
        return None
    try:
        src=_Source(units,ir_nodes)
        return _UnetExpander(src,input_shape).run()
    except (StaticPathUnsupported, KeyError, IndexError, TypeError, ValueError,
            AttributeError, RecursionError, OverflowError) as exc:
        return {'schema_version':'1.0','status':'unsupported',
                'issues':[f'{type(exc).__name__}: {exc}'],'nodes':[],'edges':[]}
