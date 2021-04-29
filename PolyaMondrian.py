import numpy as np
import pandas as pd   
from collections import defaultdict
import math
import functools
from sklearn.base import BaseEstimator
import line_profiler
import random
from sklearn.ensemble import BaggingClassifier
from pyod.utils.utility import generate_indices

class MondrianPolyaNode():

    def __init__(self,parent = None,left= None,right=None,is_leaf:bool = True,min_d = None,max_d=None,cost:float=np.Inf,
            feature = None,split= None,data_ids=None,chi_left = 0,chi_right=0,rho_left=0,rho_left_c=0,rho_right=0,rho_right_c=0,mass=1,counts=0,depth=0):

        self.parent: MondrianPolyaNode = parent
        self.left: MondrianPolyaNode = left
        self.right: MondrianPolyaNode = right
        self.is_leaf: bool = is_leaf
        self.min_d = min_d
        self.max_d = max_d
        self.cost: float = cost
        self.feature = feature
        self.split = split
        self.data_ids = data_ids
        self.chi_left = chi_left
        self.chi_right = chi_right
        self.rho_left = rho_left
        self.rho_left_c = rho_left_c
        self.rho_right = rho_right
        self.rho_right_c = rho_right_c
        self.mass = mass
        self.counts = counts
        self.depth = depth

    @property
    @functools.lru_cache()
    def range_d(self):
        return self.max_d - self.min_d
    
    @property
    @functools.lru_cache()
    def sum_range_d(self):
        return self.range_d.sum()

    @property
    @functools.lru_cache()
    def obs_vol(self):
        return self.range_d.prod()

    def get_left_and_right_vol(self):
        v_node = self.obs_vol
        
        hj = self.range_d[self.feature]

        fp = v_node/hj

        v_rj0 = fp*abs(self.min_d[self.feature] - self.split)
        v_rj1 = fp*abs(self.max_d[self.feature] - self.split)

        return v_rj0,v_rj1

    

    def choose_feature(self):
        scores_cumsum = self.range_d.cumsum()
        s = scores_cumsum[-1] * np.random.rand(1)
        feat = int(np.sum(s > scores_cumsum))
        return feat

    def choose_split(self,feature):
        
        return np.random.uniform(self.min_d[feature],self.max_d[feature])

    def __str__(self): 
        return f"Feat:{self.feature},Split:{self.split},Min_d:{self.min_d},Max_d:{self.max_d},Mass={self.mass},Chi_left={self.chi_left},chi_right={self.chi_right}, rho_left = {self.rho_left}, rho_left_c = {self.rho_left_c},  rho_right = {self.rho_right}, rho_right_c = {self.rho_right_c}"

    def __repr__(self):
        return self.__str__()

class MondrianPolyaTree():
    def __init__(self,root=None,n_classes=0,n_features=0,budget=np.Inf,prior_strength=1):
        self.root = root
        self.n_features = n_features
        self.budget = budget
        self.prior_strength = prior_strength
        

    def fit(self,X):
        
        self.X_train = X.copy()
        
        self.n_features = X.shape[1]
        
      
        self.root = MondrianPolyaNode()

        self.root.min_d = np.min(self.X_train,0)
        self.root.max_d = np.max(self.X_train,0)
        self.root.data_ids = np.arange(self.X_train.shape[0])
        self.root.counts = X.shape[0]+1
        self.sample_mondrian_polya_tree()

    def get_vol_by_ids(self,ids):
        min_d = self.X_train[ids].min(0)
        max_d = self.X_train[ids].max(0)

        range_d = max_d-min_d

        return range_d.prod()
        
    def sample_mondrian_polya_tree(self):
        
        self.sample_mondrian_polya_block(self.root)

    def score_all(self):
        scores = np.zeros(self.X_train.shape[0])
        for i in range(self.X_train.shape[0]):
            scores[i] = self.score(self.X_train[i])
        
        return scores

    def score(self,sample):

        node = self.root

        mass = 1
        while True:

            if node.is_leaf:    
                return mass        

            if sample[node.feature] <= node.split: #left
                mean_chi = node.chi_left/(node.chi_right+node.chi_left)
                mass *= mean_chi

                
                if node.left.counts > 1:
                    eta_left = np.sum(np.maximum(sample-node.left.max_d,0) + np.maximum(node.left.min_d - sample,0))
                    if eta_left > 0:
                        mean_rho = node.rho_left_c/(node.rho_left_c+node.rho_left)
                        return mass*mean_rho
                    else:
                        mean_rho = node.rho_left/(node.rho_left_c+node.rho_left)
                        mass *= mean_rho
                    

                node = node.left  
            else:
                mean_chi = node.chi_right/(node.chi_right+node.chi_left)
                mass *= mean_chi

                if node.right.counts > 1:
                    eta_right = np.sum(np.maximum(sample-node.right.max_d,0) + np.maximum(node.right.min_d - sample,0))
                    if eta_right > 0:
                        mean_rho = node.rho_right_c/(node.rho_right_c+node.rho_right)
                        return mass*mean_rho
                    else:
                        mean_rho = node.rho_right/(node.rho_right_c+node.rho_right)
                        mass *= mean_rho

                node = node.right


    def sample_mondrian_polya_block(self,node):


        grow_nodes = [node]


        while grow_nodes:
            
            # print(grow_nodes)
            node = grow_nodes.pop(0)

            split_cost = np.Inf

            if node.sum_range_d != 0.0:
                split_cost = np.random.exponential(1/node.sum_range_d)
            
            parent_cost = 0

            if node is not self.root:
                parent_cost = node.parent.cost



            # print(f"parent_cost={parent_cost}, split_cost={split_cost},all_range>0={np.all(node.range_d>0)}")
            if parent_cost + split_cost < self.budget and np.all(node.range_d>0):
                # print("------------------------------------------------------------------")
                
                node.cost = parent_cost + split_cost


                feature = node.choose_feature()
            
                split = node.choose_split(feature)


                left,right = self.cut(node,feature,split)

                rho_o_l,rho_c_l = self.restrict(node,left.data_ids,False)

                node.rho_left = rho_o_l
                node.rho_left_c = rho_c_l

                rho_o_r,rho_c_r = self.restrict(node,right.data_ids,True)

                node.rho_right = rho_o_r
                node.rho_right_c = rho_c_r

                grow_nodes.extend([left,right])

                # self.sample_mondrian_polya_block(left)
                # self.sample_mondrian_polya_block(right)

            else:
                # if not np.any(node.range_d<=0.0):
                #     self.restrict(node)
                
                node.tau = self.budget
                node.is_leaf = True

            

    def cut(self,node,feature,split):
        # feature = node.choose_feature()
        
        # split = node.choose_split(feature)


        left_ids_node = np.where(self.X_train[node.data_ids,feature] <= split)[0]

        left_ids = node.data_ids[left_ids_node]
        

        right_ids_node = np.where(self.X_train[node.data_ids,feature] > split)[0]

        right_ids = node.data_ids[right_ids_node]

        
        x_left = self.X_train[left_ids]
        
        x_right = self.X_train[right_ids]

        n_left = x_left.shape[0]

        n_right = x_right.shape[0]

        left = MondrianPolyaNode(
            min_d = np.min(x_left,0),
            max_d = np.max(x_left,0),
            parent = node,
            counts = n_left,
            data_ids = left_ids,
            depth = node.depth + 1
        )
        
        right = MondrianPolyaNode(
            min_d = np.min(x_right,0),
            max_d = np.max(x_right,0),
            parent = node,
            counts = n_right,
            data_ids = right_ids,
            depth = node.depth + 1
        )

        node.feature = feature
        node.split = split
        node.right = right
        node.left = left
        node.is_leaf = False


        v_rj0,v_rj1 = node.get_left_and_right_vol()

        d = node.depth

        # print(f"v_left:{v_rj0},v_right:{v_rj1}")
        # print(f"n_left:{n_left},n_right={n_right}")

        chi_0,chi_1 = self.compute_cut_parameters(2*d,n_left,n_right,v_rj0,v_rj1)


        # print(f"chi_0={chi_0},chi_1={chi_1}")
        node.chi_left = chi_0
        node.chi_right = chi_1


        exp_val = chi_0/(chi_1+chi_0)


        # print(f"Exp_val_cut = {exp_val}")
   

        # print(f"right.mass={right.mass}")
        # print(f"left.mass={left.mass}")

        return left,right

    def restrict(self,node,ids,is_right):
        depth = node.depth

        n_obs = ids.size

        v_p = node.get_left_and_right_vol()[is_right]
        
        v_o = self.get_vol_by_ids(ids)

        # print(f"vo:{v_o}")
        v_c = v_p - v_o
        rho_obs,rho_comp = self.compute_restrict_parameters(2*depth + 1,n_obs,v_o,v_c)
        

        #beta dist first moment
        # exp_val = rho_obs/(rho_obs+rho_comp)

        # print(f"restrict_exp_val={exp_val}")

        return rho_obs, rho_comp
      
        

    def compute_cut_parameters(self,depth,n_left,n_right,v_r_left,v_r_right):
        
        chi_0 = self.prior_strength * \
                ((depth+1)**2) * \
                (v_r_left/(v_r_left+v_r_right)) + \
                n_left
        
        chi_1 = self.prior_strength * \
                ((depth+1)**2) * \
                (v_r_right/(v_r_left+v_r_right)) + \
                n_right

        return chi_0,chi_1

    def compute_restrict_parameters(self,depth,n_obs,v_obs,v_comp):
        
        # print(f"depth={depth} n_obs={n_obs} v_obs={v_obs} v_comp={v_comp}")
        rho_obs = self.prior_strength * \
                   ((depth+1)**2) * \
                   (v_obs/(v_obs+v_comp)) + \
                   n_obs

        # print(f"rho_obs={rho_obs}")
        rho_comp = self.prior_strength * \
                   ((depth+1)**2) * \
                   (v_comp/(v_obs+v_comp))
        
        # print(F"rho_comp={rho_comp}")
        return rho_obs,rho_comp

    


class MondrianPolyaForest:

    def __init__(self,n_estimators=10):
        
        self.n_estimators = n_estimators
        self.forest = [None]*n_estimators



    def fit(self,X):

        for i in range(self.n_estimators):
            print("ola")
            idx = generate_indices(np.random.RandomState(),True,X.shape[0],X.shape[0])
            X_train = X[idx]
            mpt = MondrianPolyaTree()
            mpt.fit(X_train)
            self.forest[i] = mpt
        
        print(self.forest)

        

    def score(self,X):
        score = np.zeros(X.shape[0])
        for t in self.forest:
            score += t.score_all()
    
        return score/self.n_estimators

def main():


    print("hello world")


if __name__ == "__main__":
    main()