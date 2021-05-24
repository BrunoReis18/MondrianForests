import numpy as np
import pandas as pd   
from collections import defaultdict
import math
import functools
from sklearn.base import BaseEstimator
# import line_profiler
import random
from sklearn.ensemble import BaggingClassifier
from numpy.testing import assert_almost_equal
import time



class MondrianTreeClassifierNode:

    num_classes = 0


    def __init__(self,parent = None,left= None,right=None,is_leaf:bool = True,min_d = None,max_d=None,cost:float=np.Inf,
                feature = None,split= None,counts = None,data_ids=None, cnts = None,pred_prob =  None,id=""):
        self.id = id
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
    
    @property
    def left(self):
        return self._left
    
    @property
    def right(self):
        return self._right


    @right.setter
    def right(self,right):
        
        
        self._right = right

        if self._right is not None:
            self._right.id = self.id + "1"
    
    
    @left.setter
    def left(self,left):
        
        
        self._left = left

        if self._left is not None:

            self._left.id = self.id + "0"
        
        
    def choose_feature(self):
        
        scores_cumsum = self.range_d.cumsum()
        s = scores_cumsum[-1] * np.random.rand(1)
        feat = int(np.sum(s > scores_cumsum))
        return feat

    def choose_split(self,feature):
        
        return np.random.uniform(self.min_d[feature],self.max_d[feature])
    
    def is_internal(self,x):
        is_in_min_border = np.any(x==self.min_d)
        is_in_max_border = np.any(x==self.max_d)
        return not is_in_max_border and not is_in_min_border

    def is_paused(self):
        return self.counts[self.counts > 0].size < 2 or self.counts.sum() < 2 or self.sum_range_d==0

    def __str__(self): 
        return f"ID:{self.id},Feat:{self.feature},Split:{self.split}"

    def __repr__(self):
        return self.__str__()


class MondrianTreeClassifier(BaseEstimator):
    def __init__(self,root=None,n_classes=0,n_features=0,discount_factor=10.0,discount_param=0,budget=np.Inf,classes_=None):
        self.root = root
        self.n_classes = n_classes
        self.n_features = n_features
        self.discount_factor = discount_factor
        self.discount_param = discount_param
        self.budget = budget
        self.classes_ = classes_
        self.fit_called = False
    
    def fit(self,X,y):
        self.fit_called = True
        self.X_train = X.copy()
        self.y_train = y.astype(int).copy()

        self.classes_ = np.unique(y)

        

        self.n_classes = self.classes_.size
        
        # if self.n_classes < 2:
        #     self.classes_ = np.array([0,1])
        #     self.n_classes = 2

        self.n_features = X.shape[0]

  

        MondrianTreeClassifierNode.set_num_classes(self.n_classes)
        self.discount_param = self.discount_factor*self.n_features
        
        self.budget = (self.X_train.shape[0]) ** (1/(self.n_features + 2))

        # print(self.budget)
        self.root = MondrianTreeClassifierNode()
        self.root.min_d = np.min(self.X_train,0)
        self.root.max_d = np.max(self.X_train,0)
        self.root.data_ids = np.arange(self.X_train.shape[0])
        self.root.id= "e"

        self.root.counts = np.bincount(self.y_train,minlength=self.n_classes)
        
        # self.sample_mondrian_tree(self.root)

        self.sample_mondrian_tree(self.root)
        self.update_post_counts()
        self.update_pred_post()


    def partial_fit(self,X,y,classes=None):
        if self.fit_called == False:
            self.fit(X,y)
        else:
            new_ids = np.arange(self.X_train.shape[0],X.shape[0]+self.X_train.shape[0])
            self.X_train = np.append(self.X_train,X,0)
            self.y_train = np.append(self.y_train,y.astype(int),0)

            if self.root is None:
                self.fit(X,y)
            else:
                for i in range(X.shape[0]):
                    new_x= X[i]
                    new_y = int(y[i])
                    self.learn_one(new_x,new_y)
                # self.learn_one2(new_ids)


            self.update_post_counts()
            self.update_pred_post()

    # def get_params(self, deep = False):
    #     return {"root":self.root, "n_classes":self.n_classes,"n_features":self.n_features,"discount_factor":self.discount_factor,"discount_param":self.discount_param,"budget":self.budget}

    def is_mondrian_paused(self,node):

        return node.counts[node.counts > 0].size < 2 or node.counts.sum() < 2 or node.sum_range_d==0

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

    def learn_one(self,x_new,y_new):

        # if self.fit_called == False:
        
        #     self.fit(np.array([x_new]),y_new)
        # else:
            # self.X_train = np.append(self.X_train,x_new,0)
            # self.y_train = np.append(self.y_train,y_new.astype(int),0)

        self.extend_mondrian_block(self.root,x_new,y_new)
 
    def learn_one2(self,new_ids):
        self.extend_mondrian_block2(self.root,new_ids)

    def extend_mondrian_block2(self,node,new_ids):

        if len(new_ids) == 0:
            return

        X_new = self.X_train[new_ids]
        y_new = self.y_train[new_ids]

        min_d = X_new.min(0)
        max_d = X_new.max(0)

        new_extent_lower = np.maximum(0,node.min_d-min_d)
        new_extent_upper = np.maximum(0,X_new.max(0)-max_d)

        extent_sum = new_extent_lower+new_extent_upper

        cum_sum_extent = extent_sum.cumsum()

        expo_rate = cum_sum_extent[-1]

        split_cost = np.inf

        if expo_rate != 0:
            split_cost = random.expovariate(expo_rate)

        parent_cost = 0

        if node is not self.root:
            parent_cost = node.parent.cost
        
        unpause = False
        if node.is_paused():
            label_unique = np.unique(y_new)

            if not (len(label_unique == 1) and node.counts[label_unique[0]] > 0 and np.count_nonzero(node.counts) < 2) and (X_new.shape[0] + node.counts.sum() >= 2) :
                unpause = True

                

        if parent_cost + split_cost < node.cost:
             
            e_sample = np.random.rand() * expo_rate
            feature = (cum_sum_extent > e_sample).argmax()


            if np.random.rand() <= (new_extent_lower[feature] / extent_sum[feature]): # x exceeds maximum
                split = np.random.uniform(node.min_d[feature],min_d[feature])
            else: #x exceeds minimum
                split = np.random.uniform(max_d[feature],node.max_d[feature])

            new_parent = MondrianTreeClassifierNode(
                split = split,
                feature = feature,
                min_d = np.minimum(node.min_d,min_d),
                max_d = np.maximum(node.max_d,max_d),
                is_leaf = False,
                cost = parent_cost + split_cost,
                parent = node.parent,
                counts = np.bincount(y_new,minlength=self.n_classes) + node.counts
            )

            new_child = MondrianTreeClassifierNode(
                is_leaf = True,
                counts = np.bincount(y_new,minlength=self.n_classes),
                parent = new_parent,
                min_d = min_d.copy(),
                max_d = max_d.copy()

            )
          
            if node is self.root:
                self.root = new_parent
            else:
                if node.parent.left == node:
                    node.parent.left = new_parent
                else:
                    node.parent.right = new_parent

            node.parent = new_parent

            right_ids_new = np.where(self.X_train[new_ids,feature] > split)[0]
            left_ids_new = np.where(self.X_train[new_ids,feature] <= split)[0]

            if split <= node.max_d[feature]:

                
                new_parent.left = new_child
                new_parent.right = node
                
                new_child.data_ids = new_ids[left_ids_new]
                self.extend_mondrian_block2(node,new_ids[right_ids_new])

            else:

                new_parent.left = node
                new_parent.right = new_child

                new_child.data_ids = new_ids[right_ids_new]
                self.extend_mondrian_block2(node,new_ids[left_ids_new])
            
  
        else:

            node.min_d = np.minimum(min_d,node.min_d)
            node.max_d = np.maximum(max_d,node.max_d)
            node.counts = np.bincount(y_new,minlength=self.n_classes) + node.counts
            node.data_ids = np.append(node.data_ids,new_ids)
       
            # node.counts[y_new] += 1

            if not node.is_leaf:
                
                left_ids_new = np.where(self.X_train[new_ids,node.feature] <= node.split)[0]

                self.extend_mondrian_block2(node.left,new_ids[left_ids_new])
               
                right_ids_new = np.where(self.X_train[new_ids,node.feature] > node.split)[0]

                self.extend_mondrian_block2(node.right,new_ids[right_ids_new])

            else:
                if not node.is_paused():  
                    # assert unpause 
                    node.is_leaf = False 
                    self.sample_mondrian_tree(node)



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

                # new_child.data_ids = np.append(new_child.data_ids,self.X_train.shape[0]-1)

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
                # node.data_ids = np.append(node.data_ids,self.X_train.shape[0]-1)

                # if y_new.size != 0:

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

        # for i in range(n_samples):

        #     proba[i] = self.predict_proba_one(X_test[i])

        # return proba
        return self._predict_class(X_test)
        

    def _predict_class(self,X_test):

        pred_prob = np.zeros((X_test.shape[0], self.n_classes))
        p_not_sep_yet = np.ones(X_test.shape[0])

        d_idx_test = {self.root:np.arange(X_test.shape[0])}
        
        node_queue = [self.root]
        parent_cost = 0.0
        while True:
            
            try:
                node = node_queue.pop(0)
            except IndexError:
                break

            if node == self.root:
                parent_cost = 0.0
            else:
                parent_cost = node.parent.cost

            delta = node.cost - parent_cost
            
            # parent_cost = node.cost

            
            idx_test = d_idx_test[node]

            if len(idx_test) == 0:
                continue

            x = X_test[idx_test,:]

            eta = np.maximum(x-node.max_d,0).sum(1) + np.maximum(node.min_d - x,0).sum(1)

            if np.isinf(delta):
                p_j_not_sep = np.exp(0*eta)
            else:
                p_j_not_sep = np.exp(-delta*eta)

            # p_j_not_sep = np.exp(-delta*eta)

            p_j_sep = 1 - p_j_not_sep

            if np.isinf(delta):
                idx_zero  = eta == 0
                p_j_not_sep[idx_zero] = 1

                p_j_sep [idx_zero] = 0
            
            idx_non_zero = eta > 0
            idx_test_non_zero = idx_test[idx_non_zero]

            eta_non_zero = eta[idx_non_zero]

            base = self.get_prior_mean(node)

            # print(delta)
            if np.any(idx_non_zero):
                
                num_tables_k = np.minimum(node.cnts,1)
                num_customers = float(node.cnts.sum())
                num_tables = float(num_tables_k.sum())

                discount =  (eta_non_zero/(eta_non_zero+self.discount_param)) *  \
                            (np.expm1(delta*-(self.discount_param+eta_non_zero)))/ \
                            (np.expm1(-delta*eta_non_zero))
                
                discount_per_num_customers = discount /num_customers

                pred_prob_tmp = num_tables*discount_per_num_customers[:,np.newaxis]*base \
                                + node.cnts/num_customers - discount_per_num_customers[:,np.newaxis]*num_tables_k
                
                # print( p_j_sep[idx_non_zero][:,np.newaxis].shape)
                # print(p_not_sep_yet[idx_test_non_zero][:,np.newaxis].shape)
                # print(pred_prob_tmp.shape)
                pred_prob[idx_test_non_zero,:] += p_j_sep[idx_non_zero][:,np.newaxis] * p_not_sep_yet[idx_test_non_zero][:,np.newaxis]*pred_prob_tmp
                
                p_not_sep_yet[idx_test] *= p_j_not_sep


            if np.isinf(delta) and np.any(idx_zero):
                
                idx_test_zero = idx_test[idx_zero]
                pred_prob_node = node.pred_prob

            
                num_tables_k = np.minimum(node.cnts,1)
                num_customers = float(node.cnts.sum())
                num_tables = float(num_tables_k.sum())
                discount = self.compute_discount(node)

                pred_prob_node =  (node.cnts - discount*num_tables_k + discount*num_tables*base)/num_customers
            
                pred_prob[idx_test_zero,:] += p_not_sep_yet[idx_test_zero][:,np.newaxis] * pred_prob_node
            
            if node.is_leaf:
                continue
            try:
                feature = node.feature
                split = node.split 
                
                # print(feature)
                # print(split)
                # print(node.is_leaf)
                cond = x[:,feature] <= split

                d_idx_test[node.left],d_idx_test[node.right] = idx_test[cond],idx_test[~cond]
        
                node_queue.append(node.left)
                node_queue.append(node.right)
            except KeyError:
                    pass
        return pred_prob

    def predict_proba_one(self,sample):

        parent_cost = 0.0
        p_not_sep_yet = 1.0
    
        proba_s = np.zeros(self.n_classes)

        node= self.root

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
        


    def avg_path_length(self,node_num):
        if node_num > 2:
            return (2.0 * (np.log(node_num - 1.0) + np.euler_gamma) - 2.0 * (node_num - 1.0) / node_num)
        if node_num == 2:
            return 1.0
        else:
            return 0.0


    def path_length(self,x,depth_limit=np.Inf):

        node = self.root
        depth = 0
        while True:

            if node.is_leaf or depth > depth_limit:
                depth += self.avg_path_length(node.counts.sum())
                break
            
            if x[node.feature] <= node.split:
                depth += 1
                node = node.left
            else:
                depth += 1
                node = node.right
        return depth

    

    def __str__(self):
        nodes = [self.root]


        while nodes:

            node = nodes.pop(0)
            
            print(node)
            
                # nodes.append(node.left)
            if node.right is not None:
                nodes.insert(0,node.right)
                # nodes.append(node.right)
            if node.left is not None:
                nodes.insert(0,node.left)

        return ""
   

    def __repr__(self):
        return self.__str__()

class MondrianForestClassifier:

    def __init__(self,n_estimators=10):
        self.n_estimators = n_estimators
        self.forest = [None] * n_estimators

        for i, _ in enumerate(self.forest):
             self.forest[i] = MondrianTreeClassifier()

  
        # self.forest = BaggingClassifier(base_estimator=MondrianTreeClassifier(),n_estimators=n_estimators)
        self.fit_called = False
        self.n_classes = 0
        self._supervised = False


    def fit(self,X,y):
        self.n_classes = len(np.unique(y))
        self.fit_called = True

        for i, _ in enumerate(self.forest):
            # self.forest[i] = MondrianTreeClassifier()

            self.forest[i].fit(X,y)
  
    
    def partial_fit(self,X,y,classes=None):

        if self.fit_called == False:
            self.fit(X,y)
        else:
            for t in self.forest:
                t.partial_fit(X,y)
       
    # def partial_fit(self,X,y,classes=None):
    #     if self.fit_called == False:
    #         self.fit(X,y)
    #     else:
    #         for t in self.forest:
    #             t.partial_fit(X,y)


    def learn_one(self,X_new,y_new=np.array([])):

        # for est in self.forest.estimators_:
        #     k = np.random.poisson(1)
        #     for _ in range(k):
        #         est.learn_one(X_new,y_new)

        for t in self.forest:
            t.learn_one(X_new,y_new)
       

    def predict(self,X):

        self.pred_prob_cum = np.zeros((X.shape[0],self.n_classes))

        for t in self.forest:
           self.pred_prob_cum += t.predict_proba(X)

        return (self.pred_prob_cum/self.n_estimators).argmax(1)


    def anomaly_scores(self,X_test):

        repr_tree = self.forest[0]

        approx_avg_depth = repr_tree.avg_path_length(repr_tree.root.data_ids.size)
        
        
        anomaly_scores = []
        for x in X_test:
            tot_depth = 0.0
            for t in self.forest:
                tot_depth += t.path_length(x)

            avg_depth = tot_depth / len(self.forest)
            anomaly_scores.append(
                math.pow(
                    2.0,
                    -1.0 * (avg_depth / approx_avg_depth),
                ),
            )

        return np.asarray(anomaly_scores).ravel()
    
    def score_one(self,x,threshold=0.5):
        return self.anomaly_scores([x])
        # return self.predict_using_threshold(anom_s,threshold=threshold)

    def predict_anom(self,X_test,threshold=0.477):
        anom_s = self.anomaly_scores(X_test)
        return self.predict_using_threshold(anom_s,threshold=threshold)

    def predict_using_threshold(self, anomaly_scores, threshold=0.5):

        y_pred = np.zeros((len(anomaly_scores),))
        
        y_pred[anomaly_scores <= threshold] = 0  # --> normal
        y_pred[anomaly_scores > threshold] = 1  # --> anomaly
        y_pred = y_pred.astype(int)
        return y_pred
def main():


    print("hello world")


if __name__ == "__main__":
    main()