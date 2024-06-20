import re

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import more_itertools
from sklearn.ensemble import RandomForestClassifier
from cleanlab.pruning import get_noise_indices
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.noise_generation import generate_noisy_labels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import RandomOverSampler
import os, re, argparse
import shutil
import torch.optim as optim

import numpy as np
import pandas as pd

from gensim.models import Word2Vec


max_seq_len = 50

all_train_releases = {'activemq': 'activemq-5.0.0', 'camel': 'camel-1.4.0', 'derby': 'derby-10.2.1.6',
                      'groovy': 'groovy-1_5_7', 'hbase': 'hbase-0.94.0', 'hive': 'hive-0.9.0',
                      'jruby': 'jruby-1.1', 'lucene': 'lucene-2.3.0', 'wicket': 'wicket-1.3.0-incubating-beta-1'}

all_eval_releases = {'activemq': ['activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
                     'camel': ['camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
                     'derby': ['derby-10.3.1.4', 'derby-10.5.1.1'],
                     'groovy': ['groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
                     'hbase': ['hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.10.0', 'hive-0.12.0'],
                     'jruby': ['jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
                     'lucene': ['lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
                     'wicket': ['wicket-1.3.0-beta2', 'wicket-1.5.3']}

all_releases = {'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
                'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
                'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'],
                'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
                'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'],
                'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'],
                'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
                'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
                'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}

all_projs = list(all_train_releases.keys())

file_lvl_gt = '../datasets/preprocessed_data/'

word2vec_dir = '../output/Word2Vec_model/'

doc2vec_dir = '../output/doc2Vec_model/'


def get_df(rel, is_baseline=False):
    if is_baseline:
        df = pd.read_csv('../' + file_lvl_gt + rel + ".csv")

    else:
        df = pd.read_csv(file_lvl_gt + rel + ".csv")

    df = df.fillna('')

    df = df[df['is_blank'] == False]
    df = df[df['is_test_file'] == False]

    return df


def prepare_code2d(code_list, to_lowercase=False):
    '''
        input
            code_list (list): list that contains code each line (in str format)
        output
            code2d (nested list): a list that contains list of tokens with padding by '<pad>'
    '''
    code2d = []

    for c in code_list:
        c = re.sub('\\s+', ' ', c)

        if to_lowercase:
            c = c.lower()

        token_list = c.strip().split()
        total_tokens = len(token_list)

        token_list = token_list[:max_seq_len]

        if total_tokens < max_seq_len:
            token_list = token_list + ['<pad>'] * (max_seq_len - total_tokens)

        code2d.append(token_list)

    return code2d


def get_code3d_and_label(df, to_lowercase=False):
    '''
        input
            df (DataFrame): a dataframe from get_df()
        output
            code3d (nested list): a list of code2d from prepare_code2d()
            all_file_label (list): a list of file-level label
    '''

    code3d = []
    all_file_label = []

    for filename, group_df in df.groupby('filename'):
        file_label = bool(group_df['file-label'].unique())

        code = list(group_df['code_line'])

        code2d = prepare_code2d(code, to_lowercase)
        code3d.append(code2d)

        all_file_label.append(file_label)

    return code3d, all_file_label


def get_w2v_path():
    return word2vec_dir


def get_d2v_path():
    return doc2vec_dir


def get_w2v_weight_for_deep_learning_models(word2vec_model, embed_dim):
    word2vec_weights = torch.FloatTensor(word2vec_model.wv.syn0).cuda()

    # add zero vector for unknown tokens
    word2vec_weights = torch.cat((word2vec_weights, torch.zeros(1, embed_dim).cuda()))

    return word2vec_weights


def pad_code(code_list_3d, max_sent_len, limit_sent_len=True, mode='train'):
    paded = []

    for file in code_list_3d:
        sent_list = []
        for line in file:
            new_line = line
            if len(line) > max_seq_len:
                new_line = line[:max_seq_len]
            sent_list.append(new_line)

        if mode == 'train':
            if max_sent_len - len(file) > 0:
                for i in range(0, max_sent_len - len(file)):
                    sent_list.append([0] * max_seq_len)

        if limit_sent_len:
            paded.append(sent_list[:max_sent_len])
        else:
            paded.append(sent_list)

    return paded


def get_sample_indices(X, y, mini_index, minority_class=1):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('my_log.log', mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    m_neighbors = 6
    nn = NearestNeighbors(n_neighbors=m_neighbors)
    nn.fit(X)

    minority_indices = np.where(y == minority_class)[0]

    logger.info("minority_indices len:" + str(len(mini_index)))
    safe_indices = []
    danger_indices = []
    noise_indices = []

    for index in mini_index:
        neighbors = nn.kneighbors(X[index].reshape(1, -1), return_distance=False)[0]
        neighbors = neighbors[1:]

        minority_count = sum(y[neighbor] == minority_class for neighbor in neighbors)

        if minority_count == 0:
            noise_indices.append(index)
        elif minority_count > len(neighbors) // 2:
            safe_indices.append(index)
        else:
            danger_indices.append(index)
    logger.info("safe nums:" + str(len(safe_indices)))
    logger.info("danger nums:" + str(len(danger_indices)))
    logger.info("noise nums:" + str(len(noise_indices)))
    logger.removeHandler(fh)
    fh.close()
    return safe_indices, danger_indices, noise_indices


def cl(feature, label, train_rel):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('my_log.log', mode='a')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    file_path = '/doc/doc/wuchen/psx2/'

    df = pd.read_csv(file_path + train_rel + ".csv")
    df2 = pd.read_csv(file_lvl_gt + train_rel + ".csv")
    df2 = df2.fillna('')

    df2 = df2[df2['is_blank'] == False]
    df2 = df2[df2['is_test_file'] == False]
    computePsx = []
    for filename2, group_df2 in df2.groupby('filename'):
        for filename, group_df in df.groupby('filename'):
            if filename == filename2:
                prediction_prob = group_df['prediction-prob'].unique()
                label1 = group_df['prediction-label'].unique()
                if label1[0] == 'True':
                    p1 = prediction_prob.item()
                    p0 = 1 - p1
                    computePsx.append([p0, p1])
                else:
                    p0 = prediction_prob.item()
                    p1 = 1 - p0
                    computePsx.append([p0, p1])
    computePsx = np.array(computePsx)
    ordered = get_noise_indices(s=label,
                                psx=computePsx,
                                sorted_index_method='normalized_margin',
                                )

    issue_indices = []
    nice_indices = []
    minority_index = np.where(label == 1)[0]
    for i in minority_index:
        if i in ordered:
            issue_indices.append(i)
        else:
            nice_indices.append(i)

    logger.info("issue mini data num:" + str(len(issue_indices)))
    logger.info("nice mini data num:" + str(len(nice_indices)))
    logger.removeHandler(fh)
    fh.close()

    return issue_indices, nice_indices


def ada_ros(X, y, minority_index, multi=1):
    m_neighbors = 5
    nn = NearestNeighbors(n_neighbors=m_neighbors)
    nn.fit(X)

    majority_indices = np.where(y == 0)[0]
    majority_feature = X[majority_indices]
    majority_label = y[majority_indices]
    mini_feature = X[minority_index]
    mini_label = y[minority_index]

    n_samples = (len(majority_indices) - len(minority_index)) * multi
    n_samples = int(n_samples)
    ratio = []

    for index in minority_index:
        distances, neighbors = nn.kneighbors(X[index].reshape(1, -1))
        neighbors = nn.kneighbors(X[index].reshape(1, -1), return_distance=False)[0]
        neighbors = neighbors[1:]

        majority_ratio = sum(y[neighbor] == 0 for neighbor in neighbors) / len(neighbors)
        ratio.append(majority_ratio)

    ratio = np.array(ratio)
    ratio /= np.sum(ratio)
    n_samples_generate = np.rint(ratio * n_samples).astype(int)
    copy_index = 0
    mini_feature = mini_feature.tolist()
    mini_label = mini_label.tolist()
    for i in minority_index:
        copy = n_samples_generate[copy_index]
        for _ in range(copy):
            mini_feature.append(X[i])
            mini_label.append(y[i])
        copy_index += 1
    return n_samples_generate


def get_dataloader(code_vec, label_list, batch_size, max_sent_len, train_rel, ad=True):
    code_vec_pad = pad_code(code_vec, max_sent_len)
    majority_nums = label_list.count(False)
    minority_nums = label_list.count(True)
    imblance_ratio = majority_nums // minority_nums

    df = pd.read_csv(file_lvl_gt + train_rel + ".csv")
    df = df.fillna('')
    df = df[df['is_blank'] == False]
    df = df[df['is_test_file'] == False]

    if ad and imblance_ratio > 2:
        dim1, dim2, dim3 = len(code_vec_pad), len(code_vec_pad[0]), len(code_vec_pad[0][0])
        label_list = np.array(label_list)
        label = label_list.astype(int)
        code_vec_pad1 = np.array(code_vec_pad)
        feature = code_vec_pad1.reshape(dim1, -1)
        issue_index, nice_index = cl(feature, label, train_rel)
        safe_indices, danger_indices, noise_indices = get_sample_indices(feature, label, issue_index, minority_class=1)

        ros_index = danger_indices + noise_indices
        n_samples_generate= ada_ros(feature, label, ros_index)
        new_df = pd.DataFrame()
        i=0
        j=0
        root_dir = "/doc/doc/wuchen/experiment/RQ1/1.0/DeepLineDP-master/script/line-level-baseline/n_gram_data/"
        source_dir=root_dir+train_rel+"/line_num/"
        for root, _, files in os.walk(source_dir):
            for filename in files:
                if filename.endswith('.txt'):
                    source_file = os.path.join(root, filename)
                    if i in ros_index:
                        num_copies = n_samples_generate[j]
                        for n in range(num_copies):
                            new_filename = f"{filename}_{n + 1}.txt"
                            dest_file = os.path.join(source_dir, new_filename)
                            shutil.copyfile(source_file, dest_file)
                        j = j + 1
                    i = i + 1
                    



def get_x_vec(code_3d, word2vec):
    x_vec = [
        [[word2vec.wv.vocab[token].index if token in word2vec.wv.vocab else len(word2vec.wv.vocab) for token in text]
         for text in texts] for texts in code_3d]

    return x_vec




os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

arg = argparse.ArgumentParser()

arg.add_argument('-dataset', type=str, default='activemq', help='software project name (lowercase)')
args = arg.parse_args()
max_train_LOC = 900
file_lvl_gt = '../datasets/preprocessed_data/'
batch_size = 32
embed_dim = 50


def train_model(dataset_name):

    train_rel = all_train_releases[dataset_name]
    train_df = get_df(train_rel)

    train_code3d, train_label = get_code3d_and_label(train_df, True)
    w2v_dir = get_w2v_path()
    word2vec_file_dir = os.path.join(w2v_dir, dataset_name + '-' + str(embed_dim) + 'dim.bin')
    word2vec = Word2Vec.load(word2vec_file_dir)
    print('load Word2Vec for', dataset_name, 'finished')
    x_train_vec = get_x_vec(train_code3d, word2vec)
    max_sent_len = min(max([len(sent) for sent in (x_train_vec)]), max_train_LOC)
    get_dataloader(x_train_vec, train_label, batch_size, max_sent_len, train_rel, ad=True)



dataset_name = args.dataset
for dataset_name in ["activemq", "camel", "derby", "groovy", "hbase", "hive", "jruby", "lucene", "wicket"]:
    train_model(dataset_name)