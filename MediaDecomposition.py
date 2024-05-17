from copy import deepcopy
from typing import List
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from Definitions import *
from BayesRegression import BernoulliRegression as BReg
from BayesRegression import CompareModels


class ModelSpec:
    spec: dict = None
    name: str = None
    
    def __init__(self, name=None): 
        self.name = name
        self.spec = dict()
        
    def __str__(self): 
        return str(self.ToDict())

    def __repr__(self): 
        return self.__str__()
    
    def ToDict(self) -> dict: 
        return self.spec
    
    @staticmethod
    def __ReadInput(inp):
        if (inp is None) or (inp == []): 
            return None
        elif isinstance(inp, str):
            return [inp]
        elif isinstance(inp, list):
            return list(inp)
        else: 
            raise ValueError("Unknown input {}".format(inp))
    
    def FromLists(self, targets, media, non_media, relevance_groups, report_splits):
        self.spec[TARGETS] = ModelSpec.__ReadInput(targets)
        self.spec[MEDIA] = ModelSpec.__ReadInput(media)
        self.spec[NON_MEDIA] = ModelSpec.__ReadInput(non_media)
        self.spec[RELEVANCE_GROUPS] = ModelSpec.__ReadInput(relevance_groups)
        self.spec[REPORT_SPLITS] = ModelSpec.__ReadInput(report_splits)
        return self
    
    def FromDict(self, spec: dict):
        for section in SPEC_SECTIONS:
            if section in spec.keys():
                self.spec[section] = ModelSpec.__ReadInput(spec[section])
            else:
                print("Warning! No {} section in spec".format(section))
        return self

    def ValidateSpecVsData(self, data: pd.DataFrame) -> bool:
        for _, section in self.spec.items():
            if section is not None:
                for v in section:
                    assert v in data, "Variable {} not found in data".format(v)
        return True
    
    def Targets(self) -> list[str]:
        # список переменных, даже если одна
        return self.spec[TARGETS]
    
    def Media(self) -> list[str]:
        # список переменных, даже если одна
        return self.spec[MEDIA]
    
    def NonMedia(self) -> list[str]:
        # список переменных, даже если одна
        return self.spec[NON_MEDIA]
    
    def SetRelevanceGroup(self, relevance_group):
        self.spec[RELEVANCE_GROUPS] = ModelSpec.__ReadInput(relevance_group)
        return self
    
    def RelevanceGroup(self) -> str | None:
        # одна переменная, строка
        if self.spec[RELEVANCE_GROUPS] is None:
            return None
        else:
            return self.spec[RELEVANCE_GROUPS][0]
    
    def AllRelevanceGroups(self) -> list[str]:
        # список переменных, даже если одна
        return self.spec[RELEVANCE_GROUPS]

    def ReportSplits(self) -> list[str]:
        # список переменных, даже если одна
        return self.spec[REPORT_SPLITS]
    

        

def PrepareInput(data: pd.DataFrame, inp):
    if (inp is None) or (inp == []):
        return None
    elif isinstance(inp, str) or isinstance(inp, list):
        return data[inp].values
    else:
        raise ValueError("Some shit in input {}".format(inp))






class MediaDecomposition:
    spec: ModelSpec = None
    data: pd.DataFrame = None
    models: dict
    X_media: np.array = None
    X_non_media: np.array = None
    X_split: np.array = None

    report_splits: dict = None

    def __init__(self) -> None:
        self.models = dict()
        self.report_splits = dict()

    def PrepareModelInputs(self, spec: ModelSpec, data: pd.DataFrame):
        self.X_media = PrepareInput(data, spec.Media())
        self.X_non_media = PrepareInput(data, spec.NonMedia())
        
        # LabelEncoder нужен потому что данные могут быть 1,2,... а нужно 0,1,...
        self.X_split = PrepareInput(data, spec.RelevanceGroup())
        if self.X_split is not None: 
            self.X_split = LabelEncoder().fit_transform(self.X_split)
        
        if spec.ReportSplits() is not None:
            ohe = OneHotEncoder(sparse_output=False).fit(data[spec.ReportSplits()])
            self.report_splits['Total'] = pd.Series([True] * len(data))
            self.report_splits.update(
                dict(pd.DataFrame(
                        ohe.transform(data[spec.ReportSplits()]).astype(bool), 
                        columns=ohe.get_feature_names_out()
                    ).items()
                )
            )

    def Fit(self, spec: ModelSpec, data: pd.DataFrame, show_traces=False):
        spec.ValidateSpecVsData(data)

        self.models.clear()
        self.report_splits.clear()
        self.spec = deepcopy(spec)
        self.data = data.copy()

        self.PrepareModelInputs(spec, data)
    
        for t in spec.Targets():
            model_name = (t if self.spec.name is None else "{} {}".format(t, self.spec.name))
            y = PrepareInput(data, t)
            self.models[t] = BReg(model_name).Fit(self.X_media, self.X_non_media, self.X_split, y=y, show_trace=show_traces)
        
        return self
    
    def Contributions(self):
        #

        # dims: (resps, model_elements, targets)
        contribs_all_targets = np.stack(
            [self.models[t].Contributions(self.X_media, self.X_non_media, self.X_split) for t in self.spec.Targets()], 
            axis=-1
        )
        data_all_targets = self.data[self.spec.Targets()]

        result = {}
        report_index = ['Base', 'Non-media'] + self.spec.Media()
        report_columns = self.spec.Targets()
        
        for name, filt in self.report_splits.items():
            rep = pd.DataFrame(contribs_all_targets[filt].mean(axis=0), index=report_index, columns=report_columns)
            rep[rep < 0] = 0 
            obs = data_all_targets[filt].mean()
            rep = rep / (rep.sum() / obs)
            rep.loc["Observed", :] = obs
            result[name] = rep
        
        return pd.concat(result)
    
    def GetModel(self, target):
        if target not in self.models.keys():
            return None
        return self.models[target]
         

class ModelBuildUtils:

    def __init__(self) -> None:
        pass

    def ValidateNonMedia(self, spec: ModelSpec, data: pd.DataFrame, max_variables=5): 
        spec.ValidateSpecVsData(data)
        
        # 1. candidates sets
        X = data[spec.NonMedia()]
        feature_sets = set()
        for t in spec.Targets():
            y = data[t]
            c = 0.001
            while c < 0.5:
                model = SelectFromModel(
                    LogisticRegression(C=c, penalty="l1", dual=False, solver='liblinear').fit(X, y), 
                    prefit=True).fit(X, y)
                new_feature_set = model.get_feature_names_out()
                if (0 < len(new_feature_set)) and (len(new_feature_set) < max_variables+1):
                    feature_sets.add(tuple(new_feature_set))
                c += 0.005
        print(feature_sets)

        # 2. candidate models and compare
        Xm = PrepareInput(data, spec.Media())
        for t in spec.Targets():
            y = PrepareInput(data, t)
            models = []
            for fset in feature_sets:
                Xnm = PrepareInput(data, list(fset))
                models.append(
                    BReg(str(fset)).Fit(media=Xm, non_media=Xnm, split=None, y=y)
                )
            CompareModels(*models)

    def CompareSpecs(self, specs: List[ModelSpec], data: pd.DataFrame):
        assert len(specs) > 1, "Makes sense only for several specs"
        
        models = list()
        all_targets = set()
        for spec in specs:
            all_targets.update(spec.Targets())
            models.append(MediaDecomposition().Fit(spec, data))

        for t in all_targets:
            models_to_compare = [m.GetModel(t) for m in models if m.GetModel(t) is not None]
            CompareModels(*models_to_compare)



    def ValidateRelevanceGroups(self, spec: ModelSpec, data: pd.DataFrame):
        assert len(spec.AllRelevanceGroups()) > 1, "Makes sense only for several RG"

        rg_to_check = [None] + spec.AllRelevanceGroups()
        self.CompareSpecs(
            [ModelSpec(rg).FromDict(spec.ToDict()).SetRelevanceGroup(rg) for rg in rg_to_check], 
            data
        )

