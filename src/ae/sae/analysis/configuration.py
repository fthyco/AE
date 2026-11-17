"""Safe static configuration ingestion (JSON/TOML/YAML). No target execution."""
from __future__ import annotations
import datetime
import json
import math
import tomllib
import yaml
from ..ir.canonical import stable_id

class UniqueSafeLoader(yaml.SafeLoader):
    def construct_mapping(self,node,deep=False):
        self.flatten_mapping(node)
        result={}
        for key_node,value_node in node.value:
            key=self.construct_object(key_node,deep=deep)
            if not isinstance(key,str) or key in result:
                raise yaml.constructor.ConstructorError('while parsing mapping',node.start_mark,
                    'duplicate or non-string YAML key',key_node.start_mark)
            result[key]=self.construct_object(value_node,deep=deep)
        return result

def _parse(file):
    name=file.path.lower()
    if name.endswith('.json'):
        return json.loads(file.data,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(f'Non-finite JSON value: {x}')))
    if name.endswith('.toml'):
        return tomllib.loads(file.data.decode('utf-8-sig'))
    loader=UniqueSafeLoader(file.data.decode('utf-8-sig'))
    try:
        if not loader.check_data(): raise ValueError('Empty YAML document')
        value=loader.get_data()
        if loader.check_data(): raise ValueError('Multiple YAML documents')
        return value
    finally: loader.dispose()

def analyze_configuration(graph,files):
    for file in files:
        if not file.path.lower().endswith(('.json','.toml','.yml','.yaml')): continue
        try: data=_parse(file)
        except (ValueError,UnicodeError,RecursionError,yaml.YAMLError) as exc:
            graph.issue(kind='config_parse_failure',path=file.path,message=f'{type(exc).__name__}: {exc}')
            continue
        ref={'path':file.path,'start_byte':0,'end_byte':len(file.data),
             'start_line':1,'start_column':0,'end_line':file.data.count(b'\n')+1,
             'end_column':0,'precision':'file'}
        count=0
        class Stop(Exception): pass
        def walk(value,pointer,parent,depth=0):
            nonlocal count
            if count>=20000 or depth>100:
                graph.issue(kind='config_expansion_limit',path=file.path,message=f'Expansion limit at {pointer}')
                raise Stop()
            count+=1
            ident=stable_id(file.path,'configuration',pointer)
            attrs={'pointer':pointer,'value_kind':type(value).__name__}
            if not isinstance(value,(dict,list)):
                if isinstance(value,float) and not math.isfinite(value):
                    graph.issue(kind='nonfinite_config_number',path=file.path,message=f'Nonfinite at {pointer}')
                    attrs['value_kind']='nonfinite_float';attrs['value']=str(value)
                elif isinstance(value,(str,int,float,bool)) or value is None: attrs['value']=value
                elif isinstance(value,(datetime.date,datetime.datetime,datetime.time)): attrs['value']=value.isoformat()
                else: attrs['value']=str(value)
            rule='declarative-config-parse/v1'
            graph.add_node({'id':ident,'kind':'configuration_entry','attributes':attrs,
                            'source_refs':[ref],'evidence_class':'derived','resolution':'resolved',
                            'analysis_rule':rule})
            graph.add_edge({'id':stable_id(file.path,'config-edge',parent,ident),'source':parent,
                            'target':ident,'relation':'HAS_CONFIGURATION_ENTRY',
                            'evidence_class':'derived','analysis_rule':rule,'source_refs':[ref]})
            if isinstance(value,dict):
                for key in sorted(value):
                    if not isinstance(key,str):
                        graph.issue(kind='config_invalid_mapping_key',path=file.path,message=f'Non-string key at {pointer}')
                        continue
                    walk(value[key],pointer+'/'+key.replace('~','~0').replace('/','~1'),ident,depth+1)
            elif isinstance(value,list):
                for i,child in enumerate(value): walk(child,pointer+'/'+str(i),ident,depth+1)
        try: walk(data,'',stable_id(file.path,'source-file'))
        except Stop: pass
