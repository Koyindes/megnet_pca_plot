#!/usr/bin/env python3    
# -*- coding: utf-8 -*-

import json
import sys
import random
import pandas as pd
import numpy as np
import tensorflow as tf

from pymatgen.core.structure import Structure

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns

from collections import Counter

# from megnet.models import MEGNetModel
import warnings
warnings.filterwarnings('ignore')

from keras.optimizers import adam_v2
from keras.layers import Dense, Input, Concatenate, Add, Embedding, Dropout
from megnet.layers import MEGNetLayer, Set2Set
from megnet.activations import softplus2
from keras.regularizers import l2
from keras.backend import int_shape
from keras.models import Model
from megnet.callbacks import ModelCheckpointMAE, ManualStop, ReduceLRUponNan
from megnet.data.graph import GraphBatchDistanceConvert, GraphBatchGenerator, GaussianDistance
from megnet.data.crystal import CrystalGraph
from megnet.utils.preprocessing import DummyScaler
import os
from warnings import warn
from monty.serialization import dumpfn, loadfn

# original class Graph Model from Megnet github used to defined a new Megnet model just after
class GraphModel:
    """
    Composition of keras model and converter class for transfering structure
    object to input tensors. We add methods to train the model from
    (structures, targets) pairs

    Args:
        model: (keras model)
        graph_converter: (object) a object that turns a structure to a graph,
            check `megnet.data.crystal`
        target_scaler: (object) a scaler object for converting targets, check
            `megnet.utils.preprocessing`
        metadata: (dict) An optional dict of metadata associated with the model.
            Recommended to incorporate some basic information such as units,
            MAE performance, etc.

    """

    def __init__(self,
                 model,
                 graph_converter,
                 target_scaler=DummyScaler(),
                 metadata=None,
                 **kwargs):
        self.model = model
        self.graph_converter = graph_converter
        self.target_scaler = target_scaler
        self.metadata = metadata or {}

    def __getattr__(self, p):
        return getattr(self.model, p)

    def train(self,
              train_structures,
              train_targets,
              validation_structures=None,
              validation_targets=None,
              epochs=1000,
              batch_size=128,
              verbose=1,
              callbacks=None,
              scrub_failed_structures=False,
              prev_model=None,
              lr_scaling_factor=0.5,
              patience=500,
              **kwargs):
        """
        Args:
            train_structures: (list) list of pymatgen structures
            train_targets: (list) list of target values
            validation_structures: (list) list of pymatgen structures as validation
            validation_targets: (list) list of validation targets
            epochs: (int) number of epochs
            batch_size: (int) training batch size
            verbose: (int) keras fit verbose, 0 no progress bar, 1 only at the epoch end and 2 every batch
            callbacks: (list) megnet or keras callback functions for training
            scrub_failed_structures: (bool) whether to scrub structures with failed graph computation
            prev_model: (str) file name for previously saved model
            lr_scaling_factor: (float, less than 1) scale the learning rate down when nan loss encountered
            patience: (int) patience for early stopping
            **kwargs:
        """
        train_graphs, train_targets = self.get_all_graphs_targets(train_structures, train_targets,
                                                                  scrub_failed_structures=scrub_failed_structures)
        if validation_structures is not None:
            val_graphs, validation_targets = self.get_all_graphs_targets(
                validation_structures, validation_targets, scrub_failed_structures=scrub_failed_structures)
        else:
            val_graphs = None

        self.train_from_graphs(train_graphs,
                               train_targets,
                               validation_graphs=val_graphs,
                               validation_targets=validation_targets,
                               epochs=epochs,
                               batch_size=batch_size,
                               verbose=verbose,
                               callbacks=callbacks,
                               prev_model=prev_model,
                               lr_scaling_factor=lr_scaling_factor,
                               patience=patience,
                               **kwargs
                               )

    def train_from_graphs(self,
                          train_graphs,
                          train_targets,
                          validation_graphs=None,
                          validation_targets=None,
                          epochs=1000,
                          batch_size=128,
                          verbose=1,
                          callbacks=None,
                          prev_model=None,
                          lr_scaling_factor=0.5,
                          patience=500,
                          **kwargs
                          ):

        # load from saved model
        if prev_model:
            self.load_weights(prev_model)
        is_classification = 'entropy' in self.model.loss
        monitor = 'val_acc' if is_classification else 'val_mae'
        mode = 'max' if is_classification else 'min'
        dirname = kwargs.pop('dirname', 'callback')
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if callbacks is None:
            # with this call back you can stop the model training by `touch STOP`
            callbacks = [ManualStop()]
        train_nb_atoms = [len(i['atom']) for i in train_graphs]
        train_targets = [self.target_scaler.transform(i, j) for i, j in zip(train_targets, train_nb_atoms)]

        if validation_graphs is not None:
            filepath = os.path.join(dirname, '%s_{epoch:05d}_{%s:.6f}.hdf5' % (monitor, monitor))
            val_nb_atoms = [len(i['atom']) for i in validation_graphs]
            validation_targets = [self.target_scaler.transform(i, j) for i, j in zip(validation_targets, val_nb_atoms)]
            val_inputs = self.graph_converter.get_flat_data(validation_graphs, validation_targets)

            val_generator = self._create_generator(*val_inputs,
                                                   batch_size=batch_size)
            steps_per_val = int(np.ceil(len(validation_graphs) / batch_size))
            callbacks.extend([ReduceLRUponNan(filepath=filepath,
                                              monitor=monitor,
                                              mode=mode,
                                              factor=lr_scaling_factor,
                                              patience=patience,
                                              )])
            callbacks.extend([ModelCheckpointMAE(filepath=filepath,
                                                 monitor=monitor,
                                                 mode=mode,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 val_gen=val_generator,
                                                 steps_per_val=steps_per_val,
                                                 target_scaler=self.target_scaler)])
        else:
            val_generator = None
            steps_per_val = None
        train_inputs = self.graph_converter.get_flat_data(train_graphs, train_targets)
        # check dimension match
        self.check_dimension(train_graphs[0])
        train_generator = self._create_generator(*train_inputs, batch_size=batch_size)
        steps_per_train = int(np.ceil(len(train_graphs) / batch_size))
        self.fit_generator(train_generator, steps_per_epoch=steps_per_train,
                           validation_data=val_generator, validation_steps=steps_per_val,
                           epochs=epochs, verbose=verbose, callbacks=callbacks, **kwargs)

    def check_dimension(self, graph):
        """
        Check the model dimension against the graph converter dimension
        Args:
            graph: structure graph

        Returns:

        """
        test_inp = self.graph_converter.graph_to_input(graph)
        input_shapes = [i.shape for i in test_inp]

        model_input_shapes = [int_shape(i) for i in self.model.inputs]

        def _check_match(real_shape, tensor_shape):
            if len(real_shape) != len(tensor_shape):
                return False
            matched = True
            for i, j in zip(real_shape, tensor_shape):
                if j is None:
                    continue
                else:
                    if i == j:
                        continue
                    else:
                        matched = False
            return matched

        for i, j, k in zip(['atom features', 'bond features', 'state features'],
                           input_shapes[:3], model_input_shapes[:3]):
            matched = _check_match(j, k)
            if not matched:
                raise ValueError("The data dimension for %s is %s and does not match model "
                                 "required shape of %s" % (i, str(j), str(k)))

    def get_all_graphs_targets(self, structures, targets, scrub_failed_structures=False):
        """
        Compute the graphs from structures and spit out (graphs, targets) with options to
        automatically remove structures with failed graph computations

        Args:
            structures: (list) pymatgen structure list
            targets: (list) target property list
            scrub_failed_structures: (bool) whether to scrub those failed structures

        Returns:
            graphs, targets

        """
        graphs_valid = []
        targets_valid = []

        for i, (s, t) in enumerate(zip(structures, targets)):
            try:
                graph = self.graph_converter.convert(s)
                graphs_valid.append(graph)
                targets_valid.append(t)
            except Exception as e:
                if scrub_failed_structures:
                    warn("structure with index %d failed the graph computations" % i,
                         UserWarning)
                    continue
                else:
                    raise RuntimeError(str(e))
        return graphs_valid, targets_valid

    def predict_structure(self, structure):
        """
        Predict property from structure

        Args:
            structure: pymatgen structure or molecule

        Returns:
            predicted target value
        """
        graph = self.graph_converter.convert(structure)
        return self.predict_graph(graph)

    def predict_graph(self, graph):
        """
        Predict property from graph

        Args:
            graph: a graph dictionary, see megnet.data.graph

        Returns:
            predicted target value

        """
        inp = self.graph_converter.graph_to_input(graph)
        return self.target_scaler.inverse_transform(self.predict(inp).ravel(), len(graph['atom']))

    def _create_generator(self, *args, **kwargs):
        if hasattr(self.graph_converter, 'bond_converter'):
            kwargs.update({'distance_converter': self.graph_converter.bond_converter})
            return GraphBatchDistanceConvert(*args, **kwargs)
        else:
            return GraphBatchGenerator(*args, **kwargs)

    def save_model(self, filename):
        """
        Save the model to a keras model hdf5 and a json config for additional
        converters

        Args:
            filename: (str) output file name

        Returns:
            None
        """
        self.model.save(filename)
        dumpfn(
            {
                'graph_converter': self.graph_converter,
                'target_scaler': self.target_scaler,
                'metadata': self.metadata
            },
            filename + '.json'
        )

    @classmethod
    def from_file(cls, filename):
        """
        Class method to load model from
            filename for keras model
            filename.json for additional converters

        Args:
            filename: (str) model file name

        Returns
            GraphModel
        """
        configs = loadfn(filename + '.json')
        from keras.models import load_model
        from megnet.layers import _CUSTOM_OBJECTS
        model = load_model(filename, custom_objects=_CUSTOM_OBJECTS)
        configs.update({'model': model})
        return GraphModel(**configs)

    @classmethod
    def from_url(cls, url):
        """
        Download and load a model from a URL. E.g.
        https://github.com/materialsvirtuallab/megnet/blob/master/mvl_models/mp-2019.4.1/formation_energy.hdf5

        Args:
            url: (str) url link of the model

        Returns:
            GraphModel
        """
        import urllib.request
        fname = url.split("/")[-1]
        urllib.request.urlretrieve(url, fname)
        urllib.request.urlretrieve(url + ".json", fname + ".json")
        return cls.from_file(fname)

# original MegNetModel class without the last 3 layers (Dense_32 Dense_16 and output) (see two final_vec layers at the end! )
class MEGNetModel(GraphModel):
    """
    Construct a graph network model with or without explicit atom features
    if n_feature is specified then a general graph model is assumed,
    otherwise a crystal graph model with z number as atom feature is assumed.
+
    Args:
        nfeat_edge: (int) number of bond features
        nfeat_global: (int) number of state features
        nfeat_node: (int) number of atom features
        nblocks: (int) number of MEGNetLayer blocks
        lr: (float) learning rate
        n1: (int) number of hidden units in layer 1 in MEGNetLayer
        n2: (int) number of hidden units in layer 2 in MEGNetLayer
        n3: (int) number of hidden units in layer 3 in MEGNetLayer
        nvocal: (int) number of total element
        embedding_dim: (int) number of embedding dimension
        nbvocal: (int) number of bond types if bond attributes are types
        bond_embedding_dim: (int) number of bond embedding dimension
        ngvocal: (int) number of global types if global attributes are types
        global_embedding_dim: (int) number of global embedding dimension
        npass: (int) number of recurrent steps in Set2Set layer
        ntarget: (int) number of output targets
        act: (object) activation function
        l2_coef: (float or None) l2 regularization parameter
        is_classification: (bool) whether it is a classification task
        loss: (object or str) loss function
        metrics: (list or dict) List or dictionary of Keras metrics to be evaluated by the model during training and
            testing
        dropout: (float) dropout rate
        graph_converter: (object) object that exposes a "convert" method for structure to graph conversion
        target_scaler: (object) object that exposes a "transform" and "inverse_transform" methods for transforming the
            target values
        optimizer_kwargs (dict): extra keywords for optimizer, for example clipnorm and clipvalue
    """

    def __init__(self,
                 nfeat_edge=None,
                 nfeat_global=None,
                 nfeat_node=None,
                 nblocks=3,
                 lr=1e-3,
                 n1=64,
                 n2=32,
                 n3=16,
                 nvocal=95,
                 embedding_dim=16,
                 nbvocal=None,
                 bond_embedding_dim=None,
                 ngvocal=None,
                 global_embedding_dim=None,
                 npass=3,
                 ntarget=1,
                 act=softplus2,
                 is_classification=False,
                 loss="mse",
                 metrics=None,
                 l2_coef=None,
                 dropout=None,
                 graph_converter=None,
                 target_scaler=DummyScaler(),
                 optimizer_kwargs=None,
                 dropout_on_predict=False
                 ):

        # Build the MEG Model
        model = make_megnet_model(nfeat_edge=nfeat_edge,
                                  nfeat_global=nfeat_global,
                                  nfeat_node=nfeat_node,
                                  nblocks=nblocks,
                                  n1=n1,
                                  n2=n2,
                                  n3=n3,
                                  nvocal=nvocal,
                                  embedding_dim=embedding_dim,
                                  nbvocal=nbvocal,
                                  bond_embedding_dim=bond_embedding_dim,
                                  ngvocal=ngvocal,
                                  global_embedding_dim=global_embedding_dim,
                                  npass=npass,
                                  ntarget=ntarget,
                                  act=act,
                                  is_classification=is_classification,
                                  l2_coef=l2_coef,
                                  dropout=dropout,
                                  dropout_on_predict=dropout_on_predict)

        # Compile the model with the optimizer
        loss = 'binary_crossentropy' if is_classification else loss

        opt_params = {'lr': lr}
        if optimizer_kwargs is not None:
            opt_params.update(optimizer_kwargs)
        model.compile(adam_v2.Adam(**opt_params), loss, metrics=metrics)

        if graph_converter is None:
            graph_converter = CrystalGraph(cutoff=4, bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5))

        super().__init__(model=model, target_scaler=target_scaler, graph_converter=graph_converter)


def make_megnet_model(nfeat_edge=None, nfeat_global=None, nfeat_node=None, nblocks=3,
                      n1=64, n2=32, n3=16, nvocal=95, embedding_dim=16, nbvocal=None,
                      bond_embedding_dim=None, ngvocal=None, global_embedding_dim=None,
                      npass=3, ntarget=1, act=softplus2, is_classification=False,
                      l2_coef=None, dropout=None, dropout_on_predict=False):
    """Make a MEGNet Model

    Args:
        nfeat_edge: (int) number of bond features
        nfeat_global: (int) number of state features
        nfeat_node: (int) number of atom features
        nblocks: (int) number of MEGNetLayer blocks
        n1: (int) number of hidden units in layer 1 in MEGNetLayer
        n2: (int) number of hidden units in layer 2 in MEGNetLayer
        n3: (int) number of hidden units in layer 3 in MEGNetLayer
        nvocal: (int) number of total element
        embedding_dim: (int) number of embedding dimension
        nbvocal: (int) number of bond types if bond attributes are types
        bond_embedding_dim: (int) number of bond embedding dimension
        ngvocal: (int) number of global types if global attributes are types
        global_embedding_dim: (int) number of global embedding dimension
        npass: (int) number of recurrent steps in Set2Set layer
        ntarget: (int) number of output targets
        act: (object) activation function
        l2_coef: (float or None) l2 regularization parameter
        is_classification: (bool) whether it is a classification task
        dropout: (float) dropout rate
        dropout_on_predict (bool): Whether to use dropout during prediction and training
    Returns:
        (Model) Keras model, ready to run
    """

    # Get the setting for the training kwarg of Dropout
    dropout_training = True if dropout_on_predict else None

    # Create the input blocks
    int32 = 'int32'
    if nfeat_node is None:
        x1 = Input(shape=(None,), dtype=int32)  # only z as feature
        x1_ = Embedding(nvocal, embedding_dim)(x1)
    else:
        x1 = Input(shape=(None, nfeat_node))
        x1_ = x1
    if nfeat_edge is None:
        x2 = Input(shape=(None,), dtype=int32)
        x2_ = Embedding(nbvocal, bond_embedding_dim)(x2)
    else:
        x2 = Input(shape=(None, nfeat_edge))
        x2_ = x2
    if nfeat_global is None:
        x3 = Input(shape=(None,), dtype=int32)
        x3_ = Embedding(ngvocal, global_embedding_dim)(x3)
    else:
        x3 = Input(shape=(None, nfeat_global))
        x3_ = x3
    x4 = Input(shape=(None,), dtype=int32)
    x5 = Input(shape=(None,), dtype=int32)
    x6 = Input(shape=(None,), dtype=int32)
    x7 = Input(shape=(None,), dtype=int32)
    if l2_coef is not None:
        reg = l2(l2_coef)
    else:
        reg = None

    # two feedforward layers
    def ff(x, n_hiddens=[n1, n2]):
        out = x
        for i in n_hiddens:
            out = Dense(i, activation=act, kernel_regularizer=reg)(out)
        return out

    # a block corresponds to two feedforward layers + one MEGNetLayer layer
    # Note the first block does not contain the feedforward layer since
    # it will be explicitly added before the block
    def one_block(a, b, c, has_ff=True):
        if has_ff:
            x1_ = ff(a)
            x2_ = ff(b)
            x3_ = ff(c)
        else:
            x1_ = a
            x2_ = b
            x3_ = c
        out = MEGNetLayer(
            [n1, n1, n2], [n1, n1, n2], [n1, n1, n2],
            pool_method='mean', activation=act, kernel_regularizer=reg)(
            [x1_, x2_, x3_, x4, x5, x6, x7])

        x1_temp = out[0]
        x2_temp = out[1]
        x3_temp = out[2]
        if dropout:
            x1_temp = Dropout(dropout)(x1_temp, training=dropout_training)
            x2_temp = Dropout(dropout)(x2_temp, training=dropout_training)
            x3_temp = Dropout(dropout)(x3_temp, training=dropout_training)
        return x1_temp, x2_temp, x3_temp

    x1_ = ff(x1_)
    x2_ = ff(x2_)
    x3_ = ff(x3_)
    for i in range(nblocks):
        if i == 0:
            has_ff = False
        else:
            has_ff = True
        x1_1 = x1_
        x2_1 = x2_
        x3_1 = x3_
        x1_1, x2_1, x3_1 = one_block(x1_1, x2_1, x3_1, has_ff)
        # skip connection
        x1_ = Add()([x1_, x1_1])
        x2_ = Add()([x2_, x2_1])
        x3_ = Add()([x3_, x3_1])
    # set2set for both the atom and bond
    node_vec = Set2Set(T=npass, n_hidden=n3, kernel_regularizer=reg)([x1_, x6])
    edge_vec = Set2Set(T=npass, n_hidden=n3, kernel_regularizer=reg)([x2_, x7])
    # concatenate atom, bond, and global
    out = Concatenate(axis=-1)([node_vec, edge_vec, x3_])
    if dropout:
        final_vec = Dropout(dropout)(final_vec, training=dropout_training)
    # final dense layers
    #final_vec = Dense(n2, activation=act, kernel_regularizer=reg)(final_vec)
    #final_vec = Dense(n3, activation=act, kernel_regularizer=reg)(final_vec)
    if is_classification:
        final_act = 'sigmoid'
    else:
        final_act = None
    #out = Dense(ntarget, activation=final_act)(final_vec)
    model = Model(inputs=[x1, x2, x3, x4, x5, x6, x7], outputs=out)
    return model

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    
def df_merge(df_name, df_total):
    for i in range(1, len(df_name)):
        for s in df_name[i]:
            df = df_total[s[0]]
            for j in range(1, len(s)):
                df = pd.merge(df, df_total[s[j]])
            df_total[s] = df
    return df_total
    
def de_duplication(df_name, df_total):
    for i in range(1, len(df_name)):
        for name in df_name[i]:
            for it in df_total[name]['mp_id']:
                for del_it in df_name[i-1]:
                    df_total[del_it] = df_total[del_it][df_total[del_it]['mp_id'] != it]
    
    for key in df_total.keys():
        df_total[key] = df_total[key].reset_index(drop=True)
        
    return df_total

def get_predicted_features(df, dataname):
    global model_without_last_3layers
    if len(df) == 0:
        return []
    
    r = list(range(len(df)))
    random.shuffle(r)
    
    structures = []
    for i in r:
        structures.append(Structure.from_str(df[dataname+'_structure'][i], fmt='cif'))
    
    predicted_features = [model_without_last_3layers.predict_structure(structure) for structure in structures]
    
    pca = PCA(copy=True, n_components=3, whiten=False)
    np_predicted_features = np.array(predicted_features)
    pca_predicted_features = pca.fit_transform(np_predicted_features)
    
    for i in range(len(pca_predicted_features[0])):
        norm_proj = [proj[i] for proj in pca_predicted_features]
        norm_proj = normalization(norm_proj)
        for j in range(len(pca_predicted_features)):
            pca_predicted_features[j][i] = norm_proj[j]
        
    return pca_predicted_features

def megnet_pca(df_total, col_name):
    pca_total = {}
    for name in df_total.keys():
        pca_total[name] = get_predicted_features(df_total[name], col_name[name[0]])
        
    return pca_total
        
def plt_scatter(pca_data, ax2, i, x, y, label_name):
    global colors
    if len(pca_data) != 0:
        ax2.scatter(pca_data[:,x], pca_data[:,y], cmap='YlGnBu', marker='.', c=colors[i], s=3, label=label_name)

def single_set_distribution(df_name, pca_total, x, y):
    global colors
    
    fig = plt.figure()
    outer_grid = gs.GridSpec(2, 2)
    
    inner_grid = gs.GridSpecFromSubplotSpec(2, 2,
            subplot_spec=outer_grid[0], wspace=0.0, hspace=0.0, width_ratios=[0.5, 4], height_ratios=[4, 0.5])
            
    ax = plt.Subplot(fig, inner_grid[3])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.get_yaxis().set_visible(False)
    plt_data_x = []
    for name in df_name[0]:
        if len(pca_total[name]) != 0:
            plt_data_x = plt_data_x + [i[x] for i in pca_total[name]]
    sns.distplot(plt_data_x, ax=ax, hist=False, kde_kws={"shade": True, "color": 'gray', 'facecolor': 'gray'})
    fig.add_subplot(ax)
    
    ax1 = plt.Subplot(fig, inner_grid[0])
    ax1.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.get_xaxis().set_visible(False)
    plt_data_y = []
    for name in df_name[0]:
        if len(pca_total[name]) != 0:
            plt_data_y = plt_data_y + [i[y] for i in pca_total[name]]
    sns.distplot(plt_data_y, ax=ax1, hist=False, vertical=True, kde_kws={"shade": True, "color": 'gray', 'facecolor': 'gray'})
    fig.add_subplot(ax1)
        
    ax2 = plt.Subplot(fig, inner_grid[1])
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    for i, name in enumerate(df_name[0]):
        plt_scatter(pca_total[name], ax2, i, x, y, name)
    ax2.legend()
    fig.add_subplot(ax2)
    
    for sigma in range(1,4):
        inner_grid = gs.GridSpecFromSubplotSpec(2, 2,
            subplot_spec=outer_grid[sigma], wspace=0.0, hspace=0.0, width_ratios=[0.5, 4], height_ratios=[4, 0.5])
        
        ax = plt.Subplot(fig, inner_grid[3])
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.get_yaxis().set_visible(False)
        sns.distplot(plt_data_x, ax=ax, hist=False, kde_kws={"shade": True, "color": 'gray', 'facecolor': 'gray'})
        fig.add_subplot(ax)
        
        ax1 = plt.Subplot(fig, inner_grid[0])
        ax1.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.get_xaxis().set_visible(False)
        sns.distplot(plt_data_y, ax=ax1, hist=False, vertical=True, kde_kws={"shade": True, "color": 'gray', 'facecolor': 'gray'})
        fig.add_subplot(ax1)
        
        ax2 = plt.Subplot(fig, inner_grid[1])
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        for i, name in enumerate(df_name[0]):
            plt_data_x = np.array([i[x] for i in pca_total[name]])
            plt_data_y = np.array([i[y] for i in pca_total[name]])
            confidence_ellipse(plt_data_x, plt_data_y, ax2, n_std=sigma, 
                alpha=0.1, label=name, facecolor=colors[i], zorder=0)
        ax2.legend()
        fig.add_subplot(ax2)
    
    fig.suptitle("confidence_ellipse_pca({0},{1})".format(x, y))
        
def pca_plot(df_name, pca_total, x, y):
    fig = plt.figure()
    outer_grid = gs.GridSpec(2, 2)
    
    for index, df_name_list in enumerate(df_name):
        inner_grid = gs.GridSpecFromSubplotSpec(2, 2,
            subplot_spec=outer_grid[index], wspace=0.0, hspace=0.0, width_ratios=[0.5, 4], height_ratios=[4, 0.5])
        
        ax = plt.Subplot(fig, inner_grid[3])
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.get_yaxis().set_visible(False)
        plt_data = []
        for name in df_name_list:
            if len(pca_total[name]) != 0:
                plt_data = plt_data + [i[x] for i in pca_total[name]]
        sns.distplot(plt_data, ax=ax, hist=False, kde_kws={"shade": True, "color": 'gray', 'facecolor': 'gray'})
        fig.add_subplot(ax)
        
        ax1 = plt.Subplot(fig, inner_grid[0])
        ax1.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.get_xaxis().set_visible(False)
        plt_data = []
        for name in df_name_list:
            if len(pca_total[name]) != 0:
                plt_data = plt_data + [i[y] for i in pca_total[name]]
        sns.distplot(plt_data, ax=ax1, hist=False, vertical=True, kde_kws={"shade": True, "color": 'gray', 'facecolor': 'gray'})
        fig.add_subplot(ax1)
        
        ax2 = plt.Subplot(fig, inner_grid[1])
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        for i, name in enumerate(df_name_list):
            plt_scatter(pca_total[name], ax2, i, x, y, name)
        ax2.legend()
        fig.add_subplot(ax2)
    
    fig.suptitle("megnet_pca({0},{1})".format(x, y))
    
df_total = {}
   
# df_name = [['S', 'G'], ['SG']]
# col_name = {'S': 'scan', 'G': 'gllb-sc'}

df_name = [['P', 'S', 'H', 'G'], ['PS', 'PH', 'PG', 'SH', 'SG', 'HG'], ['PSH', 'PSG', 'PHG', 'SHG'], ['PSHG']]
col_name = {'P': 'pbe', 'S': 'scan', 'H': 'hse', 'G': 'gllb-sc'}

for key, value in col_name.items():
    df_total[key] = pd.read_csv('data/{0}_cif.csv'.format(value))

df_total = df_merge(df_name, df_total)
   
df_total = de_duplication(df_name, df_total)
   
pca_total = {}

for l in df_name:
    for name in l:
        pca_total[name] = []
        
# for the Megnet model
nfeat_bond = 100
nfeat_global = 2
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
distance_converter = GaussianDistance(gaussian_centers, gaussian_width)
graph_converter = CrystalGraph(bond_converter=distance_converter, cutoff=r_cutoff)

'''
gc = CrystalGraph(bond_converter=GaussianDistance(
    np.linspace(0, 5, 100), 0.5), cutoff=4)
'''

model_without_last_3layers = MEGNetModel(100, 2, nvocal=95, graph_converter=graph_converter)
model_formation_energy = MEGNetModel.from_file('band_gap_regression.hdf5')

#replace layers
for index, layer in enumerate(model_without_last_3layers.layers):
    weights_trained_model = model_formation_energy.layers[index].get_weights()
    model_without_last_3layers.layers[index].set_weights(weights_trained_model)

pca_total = megnet_pca(df_total, col_name)

colors = ['green', 'orange', 'red', 'blue', 'gray', 'purple']

pca_plot(df_name, pca_total, 0, 1)
single_set_distribution(df_name, pca_total, 0, 1)
pca_plot(df_name, pca_total, 0, 2)
single_set_distribution(df_name, pca_total, 0, 2)
pca_plot(df_name, pca_total, 1, 2)
single_set_distribution(df_name, pca_total, 1, 2)

plt.show()

