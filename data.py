import pickle
import random
import numpy as np

L = 121
N_BATCH = 50

teams = pickle.load(open( "teams.p", "rb" ) )
dataN = len(teams)

def rand_draft():
  flip = random.random() < 0.5
  draft = random.choice(teams)
  t1, t2, wl = [int(x) for x in draft[0].split(',')], [int(x) for x in draft[1].split(',')], draft[2]
  if flip:
    return t1, t2, wl
  else:
    return t2, t1, not wl

def get_subset(t1, sub_num):
  sub_num = sub_num - 1 if random.random() < 0.5 else sub_num
  t11 = [xx for xx in t1]
  random.shuffle(t11)
  return t11[:sub_num]

# return a subset of t1, subset of t2, as input
# return (t1, wl) as output
def rand_io():
  t1, t2, wl = rand_draft()
  subset_n = random.randint(1,5)
  sub1 = get_subset(t1, subset_n)
  sub2 = get_subset(t2, subset_n)
  ret = (sub1, sub2), (t1, wl)
  if len(sub1) <= len(sub2) and len(sub1) < 5:
    return ret
  else:
    return rand_io()

def rand_data():
  input_array =  [[0.0, 0.0] for _ in range(L)]
  output_array = [[0.0, 0.0] for _ in range(L)]
  rand_hero, rand_team = rand_io()
  input_array[rand_hero] = 1.0
  for r_hero in rand_team:
    output_array[r_hero] = [1.0, 0.0]
  return input_array, output_array

def rand_datas(NN):
  inputz, outputz = [], []
  for _ in range(NN):
    ii, oo = rand_data()
    inputz.append(ii)
    outputz.append(oo)
  return np.array(inputz), np.array(outputz)

if __name__ == "__main__":
  print len(teams)
  print teams[0]
  print rand_draft()
  print rand_io()
  
