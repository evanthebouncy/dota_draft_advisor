import pickle
import random
from model import *
from data import N_BATCH
import pickle

hero_idxs, idx_heros = pickle.load(open("hero_idxs.p", "rb"))
embednet = EmbedNet(tf.Session())
embednet.load_model('./models/embednet.ckpt')

def get_picked(sub_t1, sub_t2):
  ret = []
  for i in range(N_BATCH):
    input_array =  [[0.0, 0.0] for _ in range(L)]
    for he1 in sub_t1:
      input_array[he1] = [1.0, 0.0]
    for he2 in sub_t2:
      input_array[he2] = [0.0, 1.0]
    ret.append(input_array)
  return np.array(ret)

if __name__ == '__main__':
  sub_t1 = [1,3]
  sub_t2 = [2,4]
  picked = get_picked(sub_t1, sub_t2)
  suggestion = embednet.get_suggestion(picked)[0]
  print suggestion


# for i in range(1, 121):
#   hero = get_hero(i)
#   hero_vec = list(embednet.get_embedding(hero)[0][0])
#   hero_embeddings[i] = hero_vec


