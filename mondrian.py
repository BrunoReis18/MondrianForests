import numpy as np
import pandas as pd   
from collections import defaultdict
import math
import functools
from sklearn.base import BaseEstimator
import line_profiler
import random
from sklearn.ensemble import BaggingClassifier
from numpy.testing import assert_almost_equal
import time



class MondrianTreeClassifierNode:

    num_classes = 0


    def __init__(self,parent = None,left= None,right=None,is_leaf:bool = True,min_d = None,max_d=None,cost:float=np.Inf,
                feature = None,split= None,counts = None,data_ids=None, cnts = None,pred_prob =  None):

        self.parent: MondrianTreeClassifierNode = parent
        self.left: MondrianTreeClassifierNode = left
        self.right: MondrianTreeClassifierNode = right
        self.is_leaf: bool = is_leaf
        self.min_d = min_d
        self.max_d = max_d
        self.cost: float = cost
        self.feature = feature
        self.split = split
        self.counts = counts
        self.data_ids = data_ids
        self.cnts = cnts
        self.pred_prob = pred_prob

    @classmethod
    def set_num_classes(cls,num):
        cls.num_classes = num

    @property
    @functools.lru_cache()
    def range_d(self):
        return self.max_d - self.min_d
    
    @property
    @functools.lru_cache()
    def sum_range_d(self):
        return self.range_d.sum()

    def choose_feature(self):


        
        scores_cumsum = self.range_d.cumsum()
        s = scores_cumsum[-1] * np.random.rand(1)
        feat = int(np.sum(s > scores_cumsum))
        return feat

    def choose_split(self,feature):
        
        return np.random.uniform(self.min_d[feature],self.max_d[feature])
        



class MondrianTreeClassifier(BaseEstimator):
    def __init__(self,root=None,n_classes=0,n_features=0,discount_factor=10.0,discount_param=0,budget=np.Inf,classes_=None):
        self.root = root
        self.n_classes = n_classes
        self.n_features = n_features
        self.discount_factor = discount_factor
        self.discount_param = discount_param
        self.budget = budget
        self.classes_ = classes_
    
    def fit(self,X,y):

    

        self.X_train = X.copy()
        self.y_train = y.copy()

        self.classes_ = np.unique(y)
        self.n_classes = self.classes_.size
        
        self.n_features = X.shape[0]
        MondrianTreeClassifierNode.set_num_classes(self.n_classes)
        self.discount_param = self.discount_factor*self.n_features
        
        self.root = MondrianTreeClassifierNode()
        self.root.min_d = np.min(self.X_train,0)
        self.root.max_d = np.max(self.X_train,0)
        self.root.data_ids = np.arange(self.X_train.shape[0])
        self.root.counts = np.bincount(self.y_train,minlength=self.n_classes)
        
        # self.sample_mondrian_tree(self.root)

        self.sample_mondrian_tree(self.root)
        self.update_post_counts()
        self.update_pred_post()

    def fit_2(self,X,y):

        # self.X_train = X.copy()
        # self.y_train = y.copy()
        # self.n_classes = np.unique(self.y_train).size
        # self.n_features = self.X_train.shape[1]
        # MondrianTreeClassifierNode.set_num_classes(self.n_classes)
        # self.discount_param = self.discount_factor*self.n_features
        # self.root.min_d = np.min(self.X_train,0)
        # self.root.max_d = np.max(self.X_train,0)
        # self.root.data_ids = np.arange(self.X_train.shape[0])
        # self.root.counts = np.bincount(self.y_train,minlength=self.n_classes)

        self.classes_ = np.unique(y)
        self.n_classes = self.classes_.size
        
        self.n_features = X.shape[0]
        MondrianTreeClassifierNode.set_num_classes(self.n_classes)
        self.discount_param = self.discount_factor*self.n_features
        
        
        # self.sample_mondrian_tree(self.root)

        for i in range(X.shape[0]):
            new_x= X[i]
            new_y = y[i]
            if self.root is None:
                self.root = MondrianTreeClassifierNode()
                self.root.min_d = new_x.copy()
                self.root.max_d = new_x.copy()
                self.root.counts = np.bincount([new_y],minlength=self.n_classes)
                self.X_train = np.array([new_x])
                self.y_train = np.array([new_y])
            else:
                self.learn_one(new_x,new_y)


        self.update_post_counts()
        self.update_pred_post()

    # def get_params(self, deep = False):
    #     return {"root":self.root, "n_classes":self.n_classes,"n_features":self.n_features,"discount_factor":self.discount_factor,"discount_param":self.discount_param,"budget":self.budget}

    def sample_mondrian_tree(self,node):

        grow_nodes = [node]

        while grow_nodes:

            node = grow_nodes.pop(0)

            parent_cost = 0

            if node is not self.root:
                parent_cost = node.parent.cost

            if node.counts[node.counts > 0].size < 2 or node.counts.sum() < 2 or node.sum_range_d==0:
                node.cost = np.Inf
            
            else:
                split_cost = np.random.exponential(1/node.sum_range_d)
                node.cost = parent_cost + split_cost
        
            
            if node.cost < self.budget:
                
                
                # node.cost = parent_cost + split_cost

                feature = node.choose_feature()
            
                split = node.choose_split(feature)
                
                
                left_ids_node = np.where(self.X_train[node.data_ids,feature] <= split)[0]
                left_ids = node.data_ids[left_ids_node]
                

                right_ids_node = np.where(self.X_train[node.data_ids,feature] > split)[0]
                right_ids = node.data_ids[right_ids_node]
    
                
                x_left = self.X_train[left_ids]
                x_right = self.X_train[right_ids]

                y_left = self.y_train[left_ids]
                y_right = self.y_train[right_ids]

                
                left = MondrianTreeClassifierNode(
                    min_d = np.min(x_left,0),
                    max_d = np.max(x_left,0),
                    parent = node,
                    counts = np.bincount(y_left,minlength=self.n_classes),
                    data_ids = left_ids
                )

                node.left = left
                grow_nodes.append(left)

               
                right = MondrianTreeClassifierNode(
                    min_d = np.min(x_right,0),
                    max_d = np.max(x_right,0),
                    parent = node,
                    counts = np.bincount(y_right,minlength=self.n_classes),
                    data_ids = right_ids
                )
                node.right = right
                grow_nodes.append(right)

                node.is_leaf = False
                node.feature = feature
                node.split = split

            else:
    
                node.is_leaf = True

    def extend_mondrian(self,x_new,y_new):

        self.extend_mondrian_block(self.root,x_new,y_new)

    # @profile
    def extend_mondrian_block(self,node,x_new,y_new):

        while True:
         
            new_extent_lower = np.maximum(0,node.min_d-x_new)
            new_extent_upper = np.maximum(0,x_new-node.max_d)

            extent_sum = new_extent_lower+new_extent_upper
            
            # expo_rate = np.sum(extent_sum)

            cum_sum_extent = extent_sum.cumsum()

            expo_rate = cum_sum_extent[-1]

            split_cost = np.inf

            if expo_rate != 0:
                split_cost = random.expovariate(expo_rate)

            parent_cost = 0

            if node is not self.root:
                parent_cost = node.parent.cost
        
            if parent_cost + split_cost < node.cost:
            
                e_sample = np.random.rand() * expo_rate
                feature = (cum_sum_extent > e_sample).argmax()

                
      

                if x_new[feature] > node.min_d[feature]: # x exceeds maximum
                    split = np.random.uniform(node.min_d[feature],x_new[feature])
                else: #x exceeds minimum
                    split = np.random.uniform(x_new[feature],node.max_d[feature])

                new_parent = MondrianTreeClassifierNode(
                    split = split,
                    feature = feature,
                    min_d = np.minimum(node.min_d,x_new),
                    max_d = np.maximum(node.max_d,x_new),
                    is_leaf = False,
                    cost = parent_cost + split_cost,
                    parent = node.parent,
                    counts = np.bincount([y_new],minlength=self.n_classes) + node.counts
                )

                new_child = MondrianTreeClassifierNode(
                    is_leaf = True,
                    counts = np.bincount([y_new],minlength=self.n_classes),
                    parent = new_parent,
                    min_d = x_new.copy(),
                    max_d = x_new.copy()

                )
                
                if node is self.root:
                    self.root = new_parent
                else:
                    if node.parent.left == node:
                        node.parent.left = new_parent
                    else:
                        node.parent.right = new_parent

                if x_new[feature] <= split:
                    new_parent.left = new_child
                    new_parent.right = node
                else:
                    new_parent.left = node
                    new_parent.right = new_child
                
                node.parent = new_parent

                break
            else:
                
                node.min_d = np.minimum(x_new,node.min_d)
                node.max_d = np.maximum(x_new,node.max_d)
                # node.counts = np.bincount([y_new],minlength=self.n_classes) + node.counts

                node.counts[y_new] += 1

                if node.is_leaf:
                    break

                if x_new[node.feature] < node.split:

                    node = node.left
                    # self.extend_mondrian_block(node.left,x_new,y_new)
                else:
                    node = node.right
                    # self.extend_mondrian_block(node.right,x_new,y_new)

    def predict(self,X_test):
        return self.predict_proba(X_test).argmax(1)
    
    def predict_proba(self,X_test):

        n_samples = X_test.shape[0]
        n_features = X_test.shape[1]
        
        proba = np.zeros((n_samples,self.n_classes))

        for i in range(n_samples):

            proba[i] = self._predict(X_test[i])

        return proba

    def _predict(self,sample):

        parent_cost = 0.0
        p_not_sep_yet = 1.0
        node = self.root
        proba_s = np.zeros(self.n_classes)

        while True:
            
            delta = node.cost - parent_cost
            
            parent_cost = node.cost

            eta = np.sum(np.maximum(sample-node.max_d,0) + np.maximum(node.min_d - sample,0))
            

            if np.isinf(delta):
                p_j_not_sep = 1
            else:
                p_j_not_sep = np.exp(-delta*eta)
            p_j_sep = 1 - p_j_not_sep

            base = self.get_prior_mean(node)

            if p_j_sep > 0:

                
                num_tables_k = np.minimum(node.cnts,1)
                num_customers = float(node.cnts.sum())
                num_tables = float(num_tables_k.sum())
                
                #discount is given by moment generating function of trunctated exponential with rate=eta, max=delta and t = - discount_param
                discount =  (eta/(eta+self.discount_param)) *  \
                            (np.expm1(delta*-(self.discount_param+eta)))/ \
                            (np.expm1(-delta*eta))
                
                pred_prob_tmb =  (node.cnts - discount*num_tables_k + discount*num_tables*base)/num_customers

                proba_s += p_j_sep*p_not_sep_yet*pred_prob_tmb

                p_not_sep_yet *= p_j_not_sep
            
            if node.is_leaf:
                proba_s += p_not_sep_yet*p_j_not_sep*node.pred_prob
                return proba_s
                
            else:
                if sample[node.feature] <= node.split:
                    node = node.left  
                else:
                    node = node.right

    def learn_one(self,new_x,new_y):

        self.extend_mondrian_block(self.root,new_x,new_y)

        
    def update_post_counts(self):
        
        node_list = [self.root]
        #root node

        while True:
            if not node_list:
                break

            j = node_list.pop(0)  

            if j.is_leaf:
                j.cnts = j.counts
            else:
                j.cnts = np.minimum(j.left.counts,1) + np.minimum(j.right.counts,1)
                node_list.extend([j.left,j.right])
        

    def compute_discount(self,node):
        if node is self.root:
            return np.exp(-self.discount_param * node.cost)

        return np.exp(-self.discount_param * (node.cost - node.parent.cost))
    
    def get_prior_mean(self,node):
        if node is self.root:
            return np.ones(self.n_classes)/self.n_classes
        else:
            return node.parent.pred_prob

    def update_pred_post(self):
        
        node_list = [self.root]
        #root node

        while True:
            if not node_list:
                break
            
           
            j = node_list.pop(0)  
            
            base = self.get_prior_mean(j)
            discount = self.compute_discount(j)
           
            
            num_tables_k = np.minimum(j.cnts,1)
            num_customers = float(j.cnts.sum())
            num_tables = float(num_tables_k.sum())
            j.pred_prob =  (j.cnts - discount*num_tables_k + discount*num_tables*base)/num_customers
            
            if not j.is_leaf:
                node_list.extend([j.left,j.right])
        




class MondrianForestClassifier:

    def __init__(self,n_estimators=10):
        self.forest = BaggingClassifier(base_estimator=MondrianTreeClassifier(),n_estimators=n_estimators)


    def fit(self,X,y):
        self.forest.fit(X,y)


    def predict(self,X):
        return self.forest.predict(X)

def main():


    print("hello world")


if __name__ == "__main__":
    main()