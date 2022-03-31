# from data.pathways.pathway_loader import get_pathway_files
import itertools
import logging
from os.path import join, realpath, dirname, isfile
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Activation, BatchNormalization, multiply
from keras.regularizers import l2

# from data.pathways.pathway_loader import get_pathway_files
#from data.pathways.reactome import ReactomeNetwork
from model.layers_custom import Diagonal, SparseTF
from config_path import MAP_DATA_PATH


def get_map_from_layer(layer_dict):
    pathways = layer_dict.keys()
    print 'pathways', len(pathways)
    genes = list(itertools.chain.from_iterable(layer_dict.values()))
    genes = list(np.unique(genes))
    print 'genes', len(genes)

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in layer_dict.items():
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1

    df = pd.DataFrame(mat, index=pathways, columns=genes)
    # for k, v in layer_dict.items():
    #     print k, v
    #     df.loc[k,v] = 1
    # df= df.fillna(0)
    return df.T



def get_layer_maps(genes):
    ###check if calculated map file exist###
    index=0
    mapfile=join(MAP_DATA_PATH,'gomap{}'.format(index))
    if isfile(mapfile):
        maps=[]
        while isfile(mapfile):
            gomap=pd.read_csv(mapfile,index_col=0,low_memory=False)
            maps.append(gomap)
            index+=1
            mapfile=join(MAP_DATA_PATH,'gomap{}'.format(index))
        return maps
    roots=["GO:0005575","GO:0003674","GO:0008150"]
    genemap1_path=join(MAP_DATA_PATH,'gene_uniprot_map')
    genemap2_path=join(MAP_DATA_PATH,'gene_uniprot_map_our')
    unigomap_path=join(MAP_DATA_PATH,'uni_go_map_detail.csv')
    go_rela_path=join(MAP_DATA_PATH,'GO_relation_full')
    genemap1=pd.read_csv(genemap1_path,header=None,sep=' ')
    genemap2=pd.read_csv(genemap2_path,header=None,sep=' ')
    unigomap=pd.read_csv(unigomap_path,sep=',')
    genemap={}#{gene:[uid1,uid2,...]}
    uidmap={}#{uid:[goid1,goid2,...]}
    for index, row in genemap1.iterrows():
        if row[0] in genemap:
            genemap[row[0]].append(row[1])
        else:
            genemap[row[0]]=[row[1]]
    for index, row in genemap2.iterrows():
        if row[0] in genemap:
            genemap[row[0]].append(row[1])
        else:
            genemap[row[0]]=[row[1]]

    for index, row in unigomap.iterrows():
        if row[0] in uidmap:
            uidmap[row[0]].append(row[1])
        else:
            uidmap[row[0]]=[row[1]]
    ###get GO relation map###
    count={}
    file1 = open(go_rela_path,'r')
    lines = file1.readlines()
    golist_map={}
    go_child_map={}#{parrent:{child1:rela1,...,childn:relan}}
    go_parrent_map={}
    golist=[]
    index=0
    #build go map
    for line in lines:
        col=line.split(' ')
        currid=col[0]
        nextid=col[1]
        rela=col[2].rstrip()
        if rela.startswith('alt'):
            continue
        if rela.startswith('rep'):
            continue
        if rela.startswith('con'):
            continue
        if currid == nextid:
            continue
        if currid not in go_child_map:
            go_child_map[currid] = {}
            go_child_map[currid][nextid] = rela
            if currid not in golist_map:
                golist_map[currid] = index
                index=index+1
                golist.append(currid)
            if nextid not in golist_map:
                golist_map[nextid] = index
                index=index+1
                golist.append(nextid)
            if nextid not in go_parrent_map:
                go_parrent_map[nextid] = {}
                go_parrent_map[nextid][currid] = rela
            else:
                if currid not in go_parrent_map[nextid]:
                    go_parrent_map[nextid][currid] = rela
        else:
            if nextid not in go_child_map[currid]:
                go_child_map[currid][nextid] = rela
            if nextid not in golist_map:
                golist_map[nextid]=index
                index=index+1
                golist.append(nextid)
            if nextid not in go_parrent_map:
                go_parrent_map[nextid] = {}
                go_parrent_map[nextid][currid] = rela
            else:
                if currid not in go_parrent_map[nextid]:
                    go_parrent_map[nextid][currid] = rela

    ###build level map###
    go_level_list=[set(roots)]#[{layer0 GO set}{layer1 GO set}...]
    visitedgo={}
    while True:
        currset=go_level_list[len(go_level_list)-1]
        nextset=set()
        for currid in currset:
            if currid not in visitedgo:
                visitedgo[currid]=1
                if currid not in go_parrent_map:
                    continue
                for childgo in go_parrent_map[currid]:
                    if childgo not in visitedgo:
                        if childgo in currset:
                            continue
                        nextset.add(childgo)
                    else:
                        visitedgo[childgo]=visitedgo[childgo]+1
        if len(nextset)==0:
            break
        else:
            go_level_list.append(nextset)
    #print('layer num: ', len(go_level_list))
    #for i in range(len(go_level_list)):
    #    print('layer ',i,' num: ', len(go_level_list[i]))
    ###get gene -> GO layer5 map###
    mat = np.zeros((len(genes),len(go_level_list[5])))
    from collections import deque
    for geneindex, gene in enumerate(genes):
        if gene.startswith("GUT_GENOME"):
            temp=gene[:17]
            temp=temp+"0"*(22-len(gene))
            temp=temp+gene[17:]
            gene=temp
        if gene in genemap:
            for uid in genemap[gene]:
                if uid in uidmap:
                    for goid in uidmap[uid]:
                        ### gene -> GO layer5###
                        for i in range(5,len(go_level_list)):
                            if goid in go_level_list[i]:
                                if i==5:
                                    curr_go_level_list=list(go_level_list[i])
                                    mat[geneindex,curr_go_level_list.index(goid)]=1
                                    break
                                else:
                                    search_go_list=[]
                                    if goid in go_child_map:
                                        for search_go in go_child_map[goid]:
                                            search_go_list.append(search_go)
                                    queue=deque(search_go_list)
                                    while queue:
                                        currgo=queue.popleft()
                                        for i in range(5,len(go_level_list)):
                                            if currgo in go_level_list[i]:
                                                if i==5:
                                                    curr_go_level_list=list(go_level_list[i])
                                                    mat[geneindex,curr_go_level_list.index(currgo)]=1
                                                    break
                                                else:
                                                    if currgo in go_child_map:
                                                        for nextgo in go_child_map[currgo]:
                                                            queue.append(nextgo)
    
    map0 = pd.DataFrame(mat,index=genes,columns=list(go_level_list[5]))
    maps=[]
    maps.append(map0)
    ###go layer 5 -> go layer 4 -> ... -> go layer1###
    start_level=5
    end_level=0
    curr_level=start_level
    while curr_level>end_level:
        mat = np.zeros((len(go_level_list[curr_level]),len(go_level_list[curr_level-1])))
        currrow=list(go_level_list[curr_level])
        currcol=list(go_level_list[curr_level-1])
        for i in range(len(currrow)):
            if currrow[i] in go_child_map:
                for childgo in go_child_map[currrow[i]]:
                    if childgo in go_level_list[curr_level-1]:
                        mat[i,currcol.index(childgo)]=1
        currmap=pd.DataFrame(mat,index=currrow,columns=currcol)
        maps.append(currmap)
        curr_level-=1
        logging.info('layer {} , # of edges  {}'.format(i, currmap.sum().sum()))
    for i, tranmap in enumerate(maps):
        mappath=join(MAP_DATA_PATH,'gomap{}'.format(i))
        tranmap.to_csv(mappath)
    return maps


def shuffle_genes_map(mapp):
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    # logging.info('shuffling the map')
    # mapp = mapp.T
    # np.random.shuffle(mapp)
    # mapp= mapp.T
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    logging.info('shuffling')
    ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
    logging.info('ones_ratio {}'.format(ones_ratio))
    mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
    logging.info('random map ones_ratio {}'.format(ones_ratio))
    return mapp


def get_pnet(inputs, features, genes, n_hidden_layers, direction, activation, activation_decision, w_reg,
             w_reg_outcomes, dropout, sparse, add_unk_genes, batch_normal, kernel_initializer, use_bias=False,
             shuffle_genes=False, attention=False, dropout_testing=False, non_neg=False, sparse_first_layer=True):
    feature_names = {}
    n_features = len(features)
    n_genes = len(genes)

    if not type(w_reg) == list:
        w_reg = [w_reg] * 10

    if not type(w_reg_outcomes) == list:
        w_reg_outcomes = [w_reg_outcomes] * 10

    if not type(dropout) == list:
        dropout = [w_reg_outcomes] * 10

    w_reg0 = w_reg[0]
    w_reg_outcome0 = w_reg_outcomes[0]
    w_reg_outcome1 = w_reg_outcomes[1]
    reg_l = l2
    constraints = {}
    if non_neg:
        from keras.constraints import nonneg
        constraints = {'kernel_constraint': nonneg()}
        # constraints= {'kernel_constraint': nonneg(), 'bias_constraint':nonneg() }
    decision_outcomes = []
    outcome=inputs
    # if reg_outcomes:
    # decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(0), W_regularizer=reg_l(w_reg_outcome0), **constraints)(inputs)
    decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(0), W_regularizer=reg_l(w_reg_outcome0))(
        inputs)
    # else:
    #     decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(0))(inputs)

    # testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)

    # decision_outcome = Activation( activation=activation_decision, name='o{}'.format(0))(decision_outcome)

    # first outcome layer
    # decision_outcomes.append(decision_outcome)
   
    # if reg_outcomes:
    # decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(1), W_regularizer=reg_l(w_reg_outcome1/2.), **constraints)(outcome)
    #decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(1),
    #                         W_regularizer=reg_l(w_reg_outcome1 / 2.))(outcome)
    # else:
    #     decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(1))(outcome)


    # drop2 = Dropout(dropout, name='dropout_{}'.format(0))
    #drop2 = Dropout(dropout[0], name='dropout_{}'.format(0))

    #outcome = drop2(outcome, training=dropout_testing)

    # testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)

    decision_outcome = Activation(activation=activation_decision, name='o{}'.format(1))(decision_outcome)
    decision_outcomes.append(decision_outcome)

    if n_hidden_layers > 0:
        maps = get_layer_maps(genes)
        layer_inds = range(1, len(maps))
        # if adaptive_reg:
        #     w_regs = [float(w_reg)/float(i) for i in layer_inds]
        # else:
        #     w_regs = [w_reg] * len(maps)
        # if adaptive_dropout:
        #     dropouts = [float(dropout)/float(i) for i in layer_inds]
        # else:
        #     dropouts = [dropout]*len(maps)
        print 'original dropout', dropout
        print 'dropout', layer_inds, dropout, w_reg
        w_regs = w_reg[1:]
        w_reg_outcomes = w_reg_outcomes[1:]
        dropouts = dropout[1:]
        for i, mapp in enumerate(maps[0:-1]):
            w_reg = w_regs[i]
            w_reg_outcome = w_reg_outcomes[i]
            # dropout2 = dropouts[i]
            dropout = dropouts[1]
            names = mapp.index
            # names = list(mapp.index)
            mapp = mapp.values
            if shuffle_genes in ['all', 'pathways']:
                mapp = shuffle_genes_map(mapp)
            n_genes, n_pathways = mapp.shape
            logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
            # print 'map # ones {}'.format(np.sum(mapp))
            print 'layer {}, dropout  {} w_reg {}'.format(i, dropout, w_reg)
            layer_name = 'h{}'.format(i + 1)
            if sparse:
                hidden_layer = SparseTF(n_pathways, mapp, activation=activation, W_regularizer=reg_l(w_reg),
                                        name=layer_name, kernel_initializer=kernel_initializer,
                                        use_bias=use_bias, **constraints)
            else:
                hidden_layer = Dense(n_pathways, activation=activation, W_regularizer=reg_l(w_reg),
                                     name=layer_name, kernel_initializer=kernel_initializer, **constraints)

            outcome = hidden_layer(outcome)

            if attention:
                attention_probs = Dense(n_pathways, activation='sigmoid', name='attention{}'.format(i + 1),
                                        W_regularizer=l2(w_reg))(outcome)
                outcome = multiply([outcome, attention_probs], name='attention_mul{}'.format(i + 1))

            # if reg_outcomes:
            # decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2), W_regularizer=reg_l( w_reg2/(2**i)))(outcome)
            # decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2), W_regularizer=reg_l( w_reg_outcome), **constraints)(outcome)
            decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2),
                                     W_regularizer=reg_l(w_reg_outcome))(outcome)
            # else:
            #     decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2))(outcome)
            # testing
            if batch_normal:
                decision_outcome = BatchNormalization()(decision_outcome)
            decision_outcome = Activation(activation=activation_decision, name='o{}'.format(i + 2))(decision_outcome)
            decision_outcomes.append(decision_outcome)
            drop2 = Dropout(dropout, name='dropout_{}'.format(i + 1))
            outcome = drop2(outcome, training=dropout_testing)

            feature_names['h{}'.format(i)] = names
            # feature_names.append(names)
        ###last layer: 3 GO terms->root
        ###hidden_layer = Dense(3, activation=activation, W_regularizer=reg_l(w_regs[len(maps)+1]),
        ###                             name='h{}'.format(len(maps)), kernel_initializer=kernel_initializer, **constraints)
        ###outcome = hidden_layer(outcome)
        ###if attention:
        ###    attention_probs = Dense(n_pathways, activation='sigmoid', name='attention{}'.format(len(maps) + 1),
        ###                            W_regularizer=l2(w_reg))(outcome)
        ###    outcome = multiply([outcome, attention_probs], name='attention_mul{}'.format(len(maps) + 1))
        ###if batch_normal:
        ###    decision_outcome = BatchNormalization()(decision_outcome)
        ###decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(len(maps) + 1),
        ###                             W_regularizer=reg_l(w_reg_outcome))(outcome)
        ###decision_outcomes.append(decision_outcome)
        ###drop2 = Dropout(dropout, name='dropout_{}'.format(len(maps) + 1))
        ###outcome = drop2(outcome, training=dropout_testing)
        i = len(maps)
        feature_names['h{}'.format(i - 1)] = maps[-1].index
        feature_names['h{}'.format(i)] = maps[-1].columns
        # feature_names.append(maps[-1].index)
    return outcome, decision_outcomes, feature_names

