from skimage.transform import rescale, resize
import numpy as np
import time
import pickle as pkl
import random
import tensorflow as tf

# Set random seed using TensorFlow 2.x way for reproducibility
np.random.seed(int(time.time()))
tf.random.set_seed(int(time.time()))  # For TensorFlow-related operations (if used later)

# Weight initialization methods
def norm_weight(fan_in, fan_out):
    W_bound = np.sqrt(6.0 / (fan_in + fan_out))
    return np.asarray(np.random.uniform(low=-W_bound, high=W_bound, size=(fan_in, fan_out)), dtype=np.float32)

def conv_norm_weight(nin, nout, kernel_size):
    filter_shape = (kernel_size[0], kernel_size[1], nin, nout)
    fan_in = kernel_size[0] * kernel_size[1] * nin
    fan_out = kernel_size[0] * kernel_size[1] * nout
    W_bound = np.sqrt(6. / (fan_in + fan_out))
    W = np.asarray(np.random.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=np.float32)
    return W.astype('float32')

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

# Convert the output sequence into the required PUCIT-OHUL/KHATT label's format
def decode_sequence(sequence, dictionary):
    eol_index = dictionary['\n'][0]
    #print("eol_index: ", eol_index)
    #print("sequence: ", sequence)
    while sequence and sequence[-1] == eol_index:
        sequence.pop()
    sequence.append(eol_index)
    reversed_dictionary = dict((v[0],k) for k,v in dictionary.items()) #reverse mapping from indices to characters
    return "".join([reversed_dictionary[idx] for idx in sequence if idx in reversed_dictionary])
    '''
    outstr = ''
    #worddicts_r = dict_ind(dictionary)
    worddicts_r = dict((v[0],k) for k,v in dictionary.items()) #reverse mapping from indices to characters
    #print("worddicts_r: ", worddicts_r)
    for ind in range(1, len(sequence) - 1):
        k = len(sequence) - 1 - ind	#for handling right-to-left nature of Urdu
        #outstr += worddicts_r[int(sequence[k])]
        outstr += worddicts_r[sequence[k]]
    return outstr
    '''

# Visualize the temporal attention effect using colors that transition from yellow to green
def visualize_temporally(alpha, seq_len):
    visualization_t = np.zeros((100, 800, 3), np.float32)
    vis_sum = np.zeros((100, 800, 3), np.float32)
    color_t = np.zeros((3), np.float32)

    for t in range(seq_len):
        nt = t / seq_len
        color_t[0] = (1 - nt) * 0 + nt * 255
        color_t[1] = (1 - nt) * 255 + nt * 255
        color_t[2] = (1 - nt) * 0 + nt * 0
        alpha_t = resize(alpha[t], (100, 800))
        visualization_t[:, :, 0] = alpha_t * color_t[0]
        visualization_t[:, :, 1] = alpha_t * color_t[1]
        visualization_t[:, :, 2] = alpha_t * color_t[2]
        vis_sum += visualization_t
        visualization_t_norm = (visualization_t - visualization_t.min()) / (visualization_t.max() - visualization_t.min())
    vis_sum_norm = (vis_sum - vis_sum.min()) / (vis_sum.max() - vis_sum.min())
    return vis_sum_norm

# Load dictionary from pickle file
def load_dict_picklefile(dictFile):
    with open(dictFile, 'rb') as fp:
        lexicon = pkl.load(fp)
    return lexicon, len(lexicon)#, lexicon[32]

# Load dictionary from txt file
def load_dict_txtfile(dictFile):
    with open(dictFile) as fp:
        stuff = fp.readlines()
    
    lexicon = {}
    lex_ind = {}
    for itr, line in enumerate(stuff, 1):
        w = line.strip().split()
        lexicon[w[0]] = itr
    
    return lexicon, len(lexicon)

# Generate a reverse dictionary from lexicon
def dict_ind(lexicon):
    worddicts_r = [None] * (len(lexicon) + 1)
    for vv, kk in lexicon.items():
        if kk < len(lexicon) + 1:
            worddicts_r[kk] = vv
    return worddicts_r

import pickle

def load_combined_vocabulary(vocab_pickle, unicode_pickle):
    """
    Loads two pickle files containing mappings of indices to characters and Unicode characters.

    Args:
        vocab_pickle (str): Path to the pickle file containing {index: character} mapping.
        unicode_pickle (str): Path to the pickle file containing {index: Unicode character} mapping.

    Returns:
        dict: A dictionary where each index maps to (character, unicode_char).
    """

    # Load vocabulary.pkl (Index → Character)
    with open(vocab_pickle, "rb") as f:
        vocab_map = pickle.load(f)  # {index: character}
    print("vocab_map: ", vocab_map)

    # Load vocabulary_unicode.pkl (Index → Unicode Character)
    with open(unicode_pickle, "rb") as f:
        unicode_map = pickle.load(f)  # {index: Unicode character}
    print("unicode_map: ", unicode_map)

    # Merge both dictionaries
    combined_vocab = {}
    all_indices = set(vocab_map.keys()).union(set(unicode_map.keys()))

    for index in all_indices:
        char = vocab_map.get(index, None)  # Get character, default to None
        unicode_char = unicode_map.get(index, None)  # Get Unicode character, default to None
        combined_vocab[index] = (char, unicode_char)

    return combined_vocab  # {index: (character, unicode_char)}

