import os

root_dir = os.path.expanduser("~")
root_dir = os.path.join(root_dir, "Desktop")

print_interval = 100
save_model_iter = 1000

# train_data_path = os.path.join(root_dir, "R250Project/data/twitter_url/chunked/train_*")
# eval_data_path = os.path.join(root_dir, "R250Project/data/twitter_url/chunked/val_*")
# decode_data_path = os.path.join(root_dir, "R250Project/data/twitter_url/chunked/test_*")
# # vocab_path = os.path.join(root_dir, "Reinforce-Paraphrase-Generation/data/twitter_url/vocab")
# vocab_path = os.path.join(root_dir, "R250Project/resource/woz3/woz_vocab.txt")
# # log_root = os.path.join(root_dir, "Reinforce-Paraphrase-Generation/log_twitter")
# log_root = os.path.join(root_dir, "R250Project/log_rl")
# log_root = os.path.join(root_dir, "R250Project/log_twitter")
# log_root = os.path.join(root_dir, "R250Project/log_MLE")

train_data_path = "../data/twitter_url/chunked/train_*"
eval_data_path =  "../data/twitter_url/chunked/val_*"
decode_data_path = "../data/twitter_url/chunked/test_*"
# vocab_path = os.path.join(root_dir, "Reinforce-Paraphrase-Generation/data/twitter_url/vocab")
vocab_path = "../resource/woz3/woz_vocab.txt"
# log_root = os.path.join(root_dir, "Reinforce-Paraphrase-Generation/log_twitter")
# log_root = "../log_MLE"
# log_root = "../log_dagger+"
log_root = "../mixer_dagger"






# Hyperparameters
mode = "MLE"   # other options: MLE/RL/GTI/SO/SIO/DAGGER/DAGGER*/MIXER
alpha = 1.0
beta = 1.0
k1 = 0.9999
k2 = 3000.
hidden_dim= 256
emb_dim= 128
batch_size= 40
sample_size= 4
max_enc_steps= 40
max_dec_steps= 40
beam_size= 8
min_dec_steps= 5
vocab_size= 5000

max_iterations = 20000
lr = 1e-5
pointer_gen = True
is_coverage = False
lr_coverage = 0.15
cov_loss_wt = 1.0
max_grad_norm = 2.0
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
eps = 1e-12
use_gpu = True

# Parameter for the mixer
isMixer = False
mixer_delta = 1
mixer_T =  max_dec_steps
mixer_N_XENT_step = 4000
mixer_N_XENTRL_step = 2000


# Change to DSTC9 dataset
isDSTC = True