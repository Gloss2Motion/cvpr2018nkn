import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from os import listdir
from os import makedirs
from os import path
from scipy import stats

sys.path.append("./outside-code")

from Quaternions import Quaternions


def get_bonelengths(joints):
  """original parents"""
  # parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15,
  #                     16, 3, 18, 19, 20])

  """with hands parents"""
  # parents = np.array([
  #     -1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 17, 18, 19,
  #     17, 21, 22, 17, 24, 25, 17, 27, 28, 17, 30, 31, 3, 33, 34, 35, 36, 37,
  #     38, 36, 40, 41, 36, 43, 44, 36, 46, 47, 36, 49, 50
  # ])

  """only up parents"""
  parents = np.array([
    -1, 0, 1, 2, 0, 0, 3, 6, 7, 8, 9, 10, 11, 9, 13, 14, 9, 16, 17, 9, 19, 20, 9, 22,
    23, 3, 25, 26, 27, 28, 29, 30, 28, 32, 33, 28, 35, 36, 28, 38, 39, 28, 41, 42
  ])
  c_offsets = []
  for j in range(parents.shape[0]):
    if parents[j] != -1:
      c_offsets.append(joints[:,:,j,:] - joints[:,:,parents[j],:])
    else:
      c_offsets.append(joints[:,:,j,:])
  offsets = np.stack(c_offsets, axis=2)
  return np.sqrt(((offsets)**2).sum(axis=-1))[...,1:]


def compare_bls(bl1, bl2):
  """original relbones"""
  # relbones = np.array([-1, 0, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, -1, 13, 14,
  #                      15, -1, 17, 18, 19])
  """with hands relbones"""
#   relbones = np.array([
#       -1, 0, 1, 2, 3, -1, 5, 6, 7, -1, 9, 10, 11, -1, 13, 14, 15, -1, 17, 18, -1,
#       20, 21, -1, 23, 24, -1, 26, 27, -1, 29, 30, -1, 32, 33, 34, 35, 36, 37,
#       -1, 39, 40, -1, 42, 43, -1, 45, 46, -1, 48, 49
#   ])

  """only up relbones"""
  relbones = np.array([
    -1, 0, 1, -1, -1, -1, 5, 6, 7, 8, 9, 10, -1, 12, 13, -1, 15, 16, -1, 18, 19, -1, 21, 22,
    -1, 24, 25, 26, 27, 28, 29, -1, 31, 32, -1, 34, 35, -1, 37, 38, -1, 40, 41, -1
  ])
  bl_diff = np.abs(bl1-bl2).mean()

  bl1ratios = []
  bl2ratios = []
  for j in range(len(relbones)):
    if relbones[j] != -1:
      bl1ratios.append(bl1[j]/bl1[relbones[j]])
      bl2ratios.append(bl2[j]/bl2[relbones[j]])

  blratios_diff = np.abs(np.stack(bl1ratios) - np.stack(bl2ratios)).mean()

  return bl_diff, blratios_diff


def get_height(joints):
  # return (np.sqrt(((joints[5,:]-joints[4,:])**2).sum(axis=-1)) +
  #         np.sqrt(((joints[4,:]-joints[3,:])**2).sum(axis=-1)) +
  #         np.sqrt(((joints[3,:]-joints[2,:])**2).sum(axis=-1)) +
  #         np.sqrt(((joints[2,:]-joints[1,:])**2).sum(axis=-1)) +
  #         np.sqrt(((joints[1,:]-joints[0,:])**2).sum(axis=-1)) +
  #         np.sqrt(((joints[6,:]-joints[7,:])**2).sum(axis=-1)) +
  #         np.sqrt(((joints[7,:]-joints[8,:])**2).sum(axis=-1)) +
  #         np.sqrt(((joints[8,:]-joints[9,:])**2).sum(axis=-1)))
  return (np.sqrt(((joints[3,:]-joints[2,:])**2).sum(axis=-1)) +
          np.sqrt(((joints[2,:]-joints[1,:])**2).sum(axis=-1)) +
          np.sqrt(((joints[1,:]-joints[0,:])**2).sum(axis=-1)))


def put_in_world(states):
  joints = states[:,:-4]
  root_x = states[:,-4]
  root_y = states[:,-3]
  root_z = states[:,-2]
  root_r = states[:,-1]
  joints = joints.reshape(joints.shape[:1] + (-1, 3))

  rotation = Quaternions.id(1)
  offsets = []
  translation = np.array([[0,0,0]])

  for i in range(len(joints)):
    joints[i,:,:] = rotation * joints[i]
    joints[i,:,0] = joints[i,:,0] + translation[0,0]
    joints[i,:,1] = joints[i,:,1] + translation[0,1]
    joints[i,:,2] = joints[i,:,2] + translation[0,2]
    rotation = Quaternions.from_angle_axis(
        -root_r[i], np.array([0,1,0])
    ) * rotation
    offsets.append(rotation * np.array([0,0,1]))
    translation = translation + rotation * np.array(
        [root_x[i], root_y[i], root_z[i]]
    )

  return joints[None]

""" These paths are set with respect to the training settings in the README file.
    Please change them for the models you may train."""

seq_path1 = "./results/outputs/test/Online_Retargeting_Mixamo_Cycle_Adv_with_up/"
seq_path2 = "./results/outputs/test/Online_Retargeting_Mixamo_Cycle_Adv_with_up/"
seq_path3 = "./results/outputs/test/Online_Retargeting_Mixamo_Cycle_Adv_with_up/"

files = sorted([f for f in listdir(seq_path1) if f.endswith(".npz")])

joints1 = []
joints2 = []
joints3 = []
jointsgt = []

res1 = []
res2 = []
res3 = []
labels = []
bl_diffs = []
blratio_diffs = []
inpheights = []
tgtheights = []
vels = []
filenames = []

if not path.exists("./results/quantitative/"):
  makedirs("./results/quantitative/")

for i in range(len(files)):
  filenames.append(files[i])
  from_lbl = files[i].split("from=")[1].split("to=")[0]
  to_lbl = files[i].split("to=")[1].split(".npz")[0][:-1]
  labels.append(from_lbl+"/"+to_lbl)
  gt = put_in_world(np.load(seq_path1+files[i])["gt"][0])
  gtheight = get_height(gt[0,0])
  tgtheights.append(gtheight)
  jointsgt.append(gt)

  inp = put_in_world(np.load(seq_path1+files[i])["input_"][0])
  inpheight = get_height(inp[0,0])
  inpheights.append(inpheight)

  tgtbls = get_bonelengths(gt)
  inpbls = get_bonelengths(inp)
  bl_diff, blratio_diff = compare_bls(tgtbls[0,0], inpbls[0,0])

  bl_diffs.append(bl_diff)
  blratio_diffs.append(blratio_diff)

  cres1 = put_in_world(np.load(seq_path1+files[i])["outputB_"][0])
  joints1.append(cres1)

  cres2 = put_in_world(np.load(seq_path2+files[i])["outputB_"][0])
  joints2.append(cres2)

  cres3 = put_in_world(np.load(seq_path3+files[i])["outputB_"][0])
  joints3.append(cres3)

  res1.append(1./gtheight*((cres1-gt)**2).sum(axis=-1))
  res2.append(1./gtheight*((cres2-gt)**2).sum(axis=-1))
  res3.append(1./gtheight*((cres3-gt)**2).sum(axis=-1))

  cvel = np.std(gt, axis=1)**2
  vels.append(1./gtheight*cvel)

  print(str(i+1)+" out of "+str(len(files))+" processed")

joints1 = np.concatenate(joints1)
joints2 = np.concatenate(joints2)
joints3 = np.concatenate(joints3)
jointsgt = np.concatenate(jointsgt)

f = open("./results/quantitative/result_tables_mixamo_online_with_up.txt", "w")

feet = np.array([9, 13])
f.write("###########################################################\n")
f.write("## ALL CHARACTERS.\n")
f.write("###########################################################\n")

f.write("## AE\t\t\t\t\t"+
        "{0:.2f}".format(np.concatenate(res1).mean())+"\n")
f.write("## CYCLE\t\t\t\t"+
        "{0:.2f}".format(np.concatenate(res2).mean())+"\n")
f.write("## CYCLE ADV\t\t\t\t"+
        "{0:.2f}".format(np.concatenate(res3).mean())+"\n")

for label in ["new_motion/new_character","new_motion/known_character",
              "known_motion/new_character","known_motion/known_character"]:
  f.write("###########################################################\n")
  f.write("## "+label.upper()+".\n")
  f.write("###########################################################\n")
  idxs = [i for i, j in enumerate(labels)
          if label.split("/")[0] in j.split("/")[0] and
             label.split("/")[1] in j.split("/")[1]]
  print(idxs)
  f.write("## Number of Examples: "+
          "{0:.2f}".format(len(idxs))+".\n")
  f.write("## AE\t\t\t\t\t"+
          "{0:.2f}".format(np.concatenate(res1)[idxs].mean())+
          "\n")
  f.write("## CYCLE\t\t\t\t"+
          "{0:.2f}".format(np.concatenate(res2)[idxs].mean())+
          "\n")
  f.write("## CYCLE GAN\t\t\t\t"+
          "{0:.2f}".format(np.concatenate(res3)[idxs].mean())+
          "\n")
  f.write("###########################################################\n")
  f.write("## INPUT CHARACTER HEIGHT\t\t"+
          "{0:.2f}".format(np.array(inpheights)[idxs].mean())+"\n")
  f.write("## TARGET CHARACTER HEIGHT\t\t"+
          "{0:.2f}".format(np.array(tgtheights)[idxs].mean())+"\n")
  f.write("## AVG BONE LENGTH DIFF\t\t\t"+
          "{0:.2f}".format(np.array(bl_diffs)[idxs].mean())+"\n")
  f.write("## AVG BONE LENGTH RATIO DIFF\t\t"+
           "{0:.2f}".format(np.array(blratio_diffs)[idxs].mean())+"\n")
  f.write("## AVG COORDINATE VARIANCE\t\t"+
           "{0:.2f}".format(np.array(vels)[idxs].mean())+"\n")
  f.write("###########################################################\n\n\n\n")

f.close()

print("Done.")

