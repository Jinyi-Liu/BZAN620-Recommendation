import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, hstack
from scipy.sparse import linalg
from sklearn.metrics import confusion_matrix

def parse_date(data):
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return data


def read_data(path='', start_time='2000-01-01', end_time='2016-01-01'):
    _have = parse_date(pd.read_csv(path + '/' + 'have.csv', header=None, names=['user', 'item', 'time']))
    _wish = parse_date(pd.read_csv(path + '/' + 'wish.csv', header=None, names=['user', 'item', 'time']))
    _transact = parse_date(pd.read_csv(path + '/' + 'transac.csv', header=None, names=['giver', 'receiver', 'item', 'time']))

    pairs = pd.read_csv(path + '/' + 'pairs.csv', header=None, names=['user1', 'item1', 'user2', 'item2'])
    have = _have.loc[(start_time<_have['time'])&(_have['time']<= end_time)]
    wish = _wish.loc[(start_time<_wish['time'])&(_wish['time']<= end_time)]
    transact = _transact.loc[(start_time<_transact['time'])&(_transact['time']<= end_time)]

    return have.sort_values(by=['user', 'time']).reset_index(drop=True), pairs, transact.sort_values(
        by=['time']).reset_index(drop=True), wish.sort_values(by=['user', 'time']).reset_index(drop=True)


def user_data(user, _have, _wish, _transact):
    # Return have and wish list for user.
    return _have.loc[_have['user'] == user], _wish.loc[_wish['user'] == user], _transact.loc[
        _transact['receiver'] == user]


def get_unique_users(_have, _transact, _wish, _pairs):
    temp = np.concatenate((_have['user'].unique(), _transact['giver'].unique(), _transact['receiver'].unique(), _wish['user'].unique(),
                           _pairs['user1'].unique(), _pairs['user2'].unique()))
    return np.unique(temp)


def get_unique_products(_have, _transact, _wish, _pairs):
    temp = np.concatenate((_have['item'].unique(), _transact['item'].unique(), _wish['item'].unique(),
                           _pairs['item1'].unique(), _pairs['item2'].unique()))
    return np.unique(temp)


def get_user_wish(_user, _wish):
    return _wish.loc[_wish['user'] == _user]['item'].values


def get_user_receive(_user, _transact):
    return _transact.loc[_transact['receiver'] == _user]['item'].values


def get_max_products_len(_wish, _transact):
    a = _wish.groupby('user')['item'].count().max()
    b = _transact.groupby('receiver')['item'].count().max()
    return max(a, b)


def get_coordinates(_data, _users_dict, _products_dict):
    """
    To get the coordinates to create the sparse matrix.
    :param _data:
    :param _users_dict:
    :param _products_dict:
    :return:
    """
    row = np.array([])
    col = np.array([])

    if len(_data) != 0:
        if 'receiver' in _data.columns.values:
            # Transaction data.
            _data = _data[['receiver', 'item']].values
        elif 'user1' in _data.columns.values:
            # User 1 wants item 2 and user 2 wants item 1.
            # We view that "want" as preference.
            _data = np.vstack((_data[['user1','item2']].values, _data[['user2','item1']].values))
        else:
            # Have/wish data.
            _data = _data[['user', 'item']].values
        row = np.array(list(map(_users_dict.get, _data[:, 0])))
        col = np.array(list(map(_products_dict.get, _data[:, 1])))

    return row, col


def gen_interaction_matrix(_data, _users_dict, _products_dict):
    """
    Create the user-item interaction matrix R.
    To create a sparse matrix.
    """
    _row, _col = get_coordinates(_data, _users_dict, _products_dict)
    combine = np.vstack((_row, _col))
    # If the consumer and product interacts more than once, we still view it as once for simplicity.
    # This might violate the assumption on implicit preferences.
    temp = np.unique(combine, axis=1)
    _row, _col = temp[0], temp[1]
    return coo_matrix((np.ones(len(_row)), (_row, _col)), shape=(len(_users_dict), len(_products_dict)))


def gen_vectors(_interaction_matrix, rank):
    """
    To get the singular value decomposition matrix U and V given rank.
    :param _interaction_matrix:
    :param rank:
    :return:
    """
    U, sigma, V = linalg.svds(_interaction_matrix, k=rank)
    return U, V.T


def learning_feature(_tensor, _rank, _users_dict, _products_dict):
    """
    To learn the row and column space features.
    X1 is the row stack and X2 is the column stack.
    U1 captures the row space of the tensor.
    U2 captures the columns space of the tensor.
    :param _tensor: the tensor which we want to learn features from.
    :param _rank:
    :param _users_dict:
    :param _products_dict:
    :return: P_U and P_V.
    """
    X = [gen_interaction_matrix(_dataset, _users_dict, _products_dict) for _dataset in _tensor]
    X1 = hstack(X)
    X2 = hstack([_slice.transpose() for _slice in X])
    U1, S1, V1 = linalg.svds(X1, k=_rank)
    U2, S2, V2 = linalg.svds(X2, k=_rank)
    P_users = U1 @ U1.transpose()
    P_products = U2 @ U2.transpose()
    return P_users, P_products


def evaluate_predict(_hat_M, _test_X, _theta):
    """
    To evaluate the prediction accuracy by calculating TPR and FPR w.r.t. different theta.
    :param _hat_M: the learnt slice from the tensor.
    :param _test_X: the interaction matrix we want to test.
    :param _theta: the threshold for classifying.
    :return: true positive rate and false positive rate
    """
    print(_hat_M.max())
    t_hat_M = np.where(_hat_M > _theta, 1, 0)
    c_matrix = confusion_matrix(_test_X.flatten(), t_hat_M.flatten()).ravel()
    print(c_matrix)
    tp,fp = c_matrix[3], c_matrix[1]
    if tp+fp==0:
        return 0, 1
    else:
        TPR = tp/(fp+tp)
        FPR = 1-TPR
    return TPR, FPR