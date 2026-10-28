import tensorflow as tf
from tensorflow.keras import backend as K
import glob
import random
import pretty_midi
import numpy as np
from tqdm import tnrange, tqdm_notebook, tqdm
from random import shuffle, seed
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Nadam
from numpy.random import choice
import pickle
import matplotlib.pyplot as plt
import os

# ---- added imports -------------------------------------------------------
import sys
import json
import time
import platform
from collections import Counter, defaultdict
import pandas as pd
try:
    from scipy import stats as scipy_stats
except ImportError:          # statistical tests are skipped if scipy is absent
    scipy_stats = None


# =============================================================================
# SECTION A -- CONFIG, SEEDS, OUTPUT FOLDERS
# =============================================================================
SEED = 666                                   # [R1-2][R2-3] reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- dataset ---------------------------------------------------------------
MAESTRO_DIR = r'D:\Akademik\Doktora\DL\maestro-v1.0.0'               # change to your MAESTRO folder / version
MAESTRO_CSV = os.path.join(MAESTRO_DIR, os.path.basename(MAESTRO_DIR) + '.csv')
# [R1-2][R2-3][R2-4] explicit piece-level split (pieces, not sequences, so that
# no piece leaks between train / validation / test).
N_TRAIN, N_VAL, N_TEST = 80, 10, 10

# --- representation --------------------------------------------------------
seq_len = 50                                 # input sequence length (time steps)
FRAME_PER_SECOND = 5                         # piano-roll frames per second (fs)
# [R1-5][R2-7] ONE playback rate for every condition. The original script wrote
# the random-start file with fs=7 and the one-note file with fs=8, which by
# itself changes the perceived tempo and confounds the "slower tempo" claim.
PLAYBACK_FS = FRAME_PER_SECOND               # 5 fps = the time scale of the training data

# --- model / training ------------------------------------------------------
EPOCHS = 50                                  # single definition 
                                             
BATCH_SONG = 16                              # pieces decoded per loading chunk (NOT the
                                             # number of pieces in the study) [R1-3][R2-4]
BATCH_NNET_SIZE = 96                         # sequences per gradient step      [R1-3][R2-4]
DROPOUT = 0.3
OUTPUT_EMB = 100
RNN_UNIT = 128
DENSE_UNIT = 64
ATTENTION_WIDTH = 50
ATTENTION_HEADS = 1                          # SeqSelfAttention is a single-head layer
RUN_LSTM_BASELINE = True                     # [R2-5] same pipeline, LSTM instead of GRU

# --- generation experiments ------------------------------------------------
N_RUNS = 10                                  # [R1-5] independent generations per condition
MAX_GENERATE = 300                           # [R1-5] same length for every condition
TEMPERATURE = 1.0                            # 1.0 = sampling from the raw softmax (original)
ONE_NOTE_START = '72'                        # token used for the single-note start

# --- optional heavy reports ------------------------------------------------
RUN_DATASET_REPORT = True                    # [R1-3][R2-4] loads every selected MIDI once
RUN_FPS_REPORT = True                        # [R2-4] justification of fs = 5

OUT_DIR = 'results'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
TAB_DIR = os.path.join(OUT_DIR, 'tables')
MIDI_DIR = os.path.join(OUT_DIR, 'midi')
LISTEN_DIR = os.path.join(OUT_DIR, 'listening_test')
for _d in (OUT_DIR, FIG_DIR, TAB_DIR, MIDI_DIR, LISTEN_DIR):
    os.makedirs(_d, exist_ok=True)


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):

    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):

        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def get_list_midi(folder = 'maestro-v1.0.0/**/*.midi', seed_int = 666):

  list_all_midi = glob.glob(folder)
  seed(seed_int)
  shuffle(list_all_midi)
  return list_all_midi


# -----------------------------------------------------------------------------
# [R1-2][R2-4] cache of boolean piano rolls. Results are identical to the original
# code (it only ever used `roll > 0`) but every file is decoded once instead of
# once per epoch / per report, and memory stays small (bool, not float64).
# -----------------------------------------------------------------------------
_ROLL_CACHE = {}

def get_bool_roll(midi_file_name, fs):
    key = (midi_file_name, fs)
    if key not in _ROLL_CACHE:
        midi_pretty_format = pretty_midi.PrettyMIDI(midi_file_name)
        piano_midi = midi_pretty_format.instruments[0]
        _ROLL_CACHE[key] = piano_midi.get_piano_roll(fs=fs) > 0
    return _ROLL_CACHE[key]


class NoteTokenizer:

    def __init__(self):
      self.notes_to_index = {}
      self.index_to_notes = {}
      self.num_of_word = 0
      self.unique_word = 0
      self.notes_freq = {}
      # [R2-4] out-of-vocabulary bookkeeping (validation/test chords never seen in training)
      self.n_tokens_seen = 0
      self.n_oov = 0

    def reset_oov(self):
      self.n_tokens_seen = 0
      self.n_oov = 0

    def transform(self,list_array):

      transformed_list = []
      e_idx = self.notes_to_index.get('e')
      for instance in list_array:
          row = []
          for note in instance:
              idx = self.notes_to_index.get(note)
              self.n_tokens_seen += 1
              if idx is None:
                  # [R2-3] The tokenizer is now fitted on TRAIN pieces only, so a chord
                  # that never occurs in training is mapped to the rest token 'e'.
                  # The OOV rate is reported (see tokenizer / dataset reports).
                  if e_idx is None:
                      raise KeyError(note)
                  self.n_oov += 1
                  idx = e_idx
              row.append(idx)
          transformed_list.append(row)
      return np.array(transformed_list, dtype=np.int32)

    def partial_fit(self, notes):

        for note in notes:
            note_str = ','.join(str(a) for a in note)
            if note_str in self.notes_freq:
                self.notes_freq[note_str] += 1
                self.num_of_word += 1
            else:
                self.notes_freq[note_str] = 1
                self.unique_word += 1
                self.num_of_word += 1
                self.notes_to_index[note_str], self.index_to_notes[self.unique_word] = self.unique_word, note_str

    def add_new_note(self, note):

        assert note not in self.notes_to_index
        self.unique_word += 1
        self.notes_to_index[note], self.index_to_notes[self.unique_word] = self.unique_word, note


def generate_batch_song(list_all_midi, batch_music=16, start_index=0, fs=30, seq_len=50, use_tqdm=False):
  
    assert len(list_all_midi) >= batch_music
    dict_time_notes = generate_dict_time_notes(list_all_midi, batch_music, start_index, fs, use_tqdm=use_tqdm)
    
    list_musics = process_notes_in_song(dict_time_notes, seq_len)
    collected_list_input, collected_list_target = [], []
     
    for music in list_musics:
        list_training, list_target = generate_input_and_target(music, seq_len)
        collected_list_input += list_training
        collected_list_target += list_target
    return collected_list_input, collected_list_target

def generate_dict_time_notes(list_all_midi, batch_song = 16, start_index=0, fs=30, use_tqdm=True):

    assert len(list_all_midi) >= batch_song
    
    dict_time_notes = {}
    process_tqdm_midi = tqdm_notebook(range(start_index, min(start_index + batch_song, len(list_all_midi)))) if use_tqdm else range(start_index,  min(start_index + batch_song, len(list_all_midi)))
    for i in process_tqdm_midi:
        midi_file_name = list_all_midi[i]
        if use_tqdm:
            process_tqdm_midi.set_description("Processing {}".format(midi_file_name))
        try: 
            # [R1-2] cached boolean piano roll (same content as before, see get_bool_roll)
            piano_roll = get_bool_roll(midi_file_name, fs)
            dict_time_notes[i] = piano_roll
        except Exception as e:
            print(e)
            print("broken file : {}".format(midi_file_name))
            pass
    return dict_time_notes

def generate_input_and_target(dict_keys_time, seq_len=50):
 

    start_time, end_time = list(dict_keys_time.keys())[0], list(dict_keys_time.keys())[-1]
    list_training, list_target = [], []
    for index_enum, time in enumerate(range(start_time, end_time)):
        list_append_training, list_append_target = [], []
        start_iterate = 0
        flag_target_append = False 
        if index_enum < seq_len:
            start_iterate = seq_len - index_enum - 1
            for i in range(start_iterate): 
                list_append_training.append('e')
                flag_target_append = True

        for i in range(start_iterate,seq_len):
            index_enum = time - (seq_len - i - 1)
            if index_enum in dict_keys_time:
                list_append_training.append(','.join(str(x) for x in dict_keys_time[index_enum]))      
            else:
                list_append_training.append('e')


        if time+1 in dict_keys_time:
            list_append_target.append(','.join(str(x) for x in dict_keys_time[time+1]))
        else:
            list_append_target.append('e')
        list_training.append(list_append_training)
        list_target.append(list_append_target)
    return list_training, list_target

def process_notes_in_song(dict_time_notes, seq_len = 50):
   
    list_of_dict_keys_time = []
    
    for key in dict_time_notes:
        sample = dict_time_notes[key]
        times = np.unique(np.where(sample > 0)[1])
        index = np.where(sample > 0)
        dict_keys_time = {}

        for time in times:
            index_where = np.where(index[1] == time)
            notes = index[0][index_where]
            dict_keys_time[time] = notes
        list_of_dict_keys_time.append(dict_keys_time)
    return list_of_dict_keys_time


# =============================================================================
# SECTION B -- DATA: split + reports                    [R1-2][R1-3][R2-3][R2-4]
# =============================================================================
list_all_midi = get_list_midi(os.path.join(MAESTRO_DIR, '**', '*.midi'))
if len(list_all_midi) == 0:
    raise FileNotFoundError("No .midi files found under '{}'. Set MAESTRO_DIR.".format(MAESTRO_DIR))


def split_dataset(list_all_midi, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
                  csv_path=MAESTRO_CSV, maestro_dir=MAESTRO_DIR, seed_int=SEED):
    """[R1-2][R1-3][R2-3][R2-4] Piece-level train/validation/test split.

    If the official MAESTRO csv is present its 'split' column is used (the official
    split keeps all performances of one composition in the same subset); the
    pieces inside each subset are chosen by a seeded random draw. Otherwise a
    seeded random split of the file list is used. The 'why these pieces' answer
    for the manuscript is therefore: seeded uniform random sample, N_TRAIN/N_VAL/N_TEST.
    """
    rng = random.Random(seed_int)
    norm = {os.path.normpath(p): p for p in list_all_midi}
    pools = None
    if os.path.exists(csv_path):
        meta = pd.read_csv(csv_path)
        pools = {}
        for s in ('train', 'validation', 'test'):
            names = meta.loc[meta['split'] == s, 'midi_filename']
            files = [norm[os.path.normpath(os.path.join(maestro_dir, f))] for f in names
                     if os.path.normpath(os.path.join(maestro_dir, f)) in norm]
            rng.shuffle(files)
            pools[s] = files
        if min(len(v) for v in pools.values()) == 0:
            pools = None
    if pools is not None:
        train, val, test = pools['train'][:n_train], pools['validation'][:n_val], pools['test'][:n_test]
        source = 'official MAESTRO split (csv column "split"), seeded random subset of each split'
    else:
        files = list(list_all_midi)                      # already shuffled with seed 666
        train = files[:n_train]
        val = files[n_train:n_train + n_val]
        test = files[n_train + n_val:n_train + n_val + n_test]
        source = 'seeded random split of the file list (official csv not found)'
    return train, val, test, source


train_midi, val_midi, test_midi, SPLIT_SOURCE = split_dataset(list_all_midi)
sampled_200_midi = train_midi            # old variable name kept: it is now the TRAIN list only
print("MAESTRO files found: {} | train {} / val {} / test {} | {}".format(
    len(list_all_midi), len(train_midi), len(val_midi), len(test_midi), SPLIT_SOURCE))


def verify_single_instrument(files):
    """[R2-4] Verifies the claim 'the files contain a single instrument' and reports the
    duration / note count of every selected piece. The code uses instruments[0]."""
    rows = []
    for f in files:
        try:
            pm_ = pretty_midi.PrettyMIDI(f)
            rows.append({'file': os.path.basename(f),
                         'n_instruments': len(pm_.instruments),
                         'n_drum_tracks': int(sum(i.is_drum for i in pm_.instruments)),
                         'programs': sorted({int(i.program) for i in pm_.instruments}),
                         'duration_s': float(pm_.get_end_time()),
                         'n_notes': int(sum(len(i.notes) for i in pm_.instruments))})
        except Exception as e:
            rows.append({'file': os.path.basename(f), 'n_instruments': -1, 'error': str(e)})
    return pd.DataFrame(rows)


def dataset_report(split_files):
    """[R1-3][R2-4] How many pieces / hours / notes are really used, per split."""
    rows, per_file = [], []
    for split_name, files in split_files.items():
        df = verify_single_instrument(files)
        df['split'] = split_name
        per_file.append(df)
        ok = df[df['n_instruments'] >= 0]
        rows.append({'split': split_name,
                     'n_pieces': len(files),
                     'hours': round(float(ok['duration_s'].sum()) / 3600, 3),
                     'n_notes': int(ok['n_notes'].sum()),
                     'share_single_instrument': round(float((ok['n_instruments'] == 1).mean()), 3),
                     'maestro_files_in_folder': len(list_all_midi)})
    per_file = pd.concat(per_file, ignore_index=True)
    summary = pd.DataFrame(rows)
    per_file.to_csv(os.path.join(TAB_DIR, 'dataset_pieces.csv'), index=False)
    summary.to_csv(os.path.join(TAB_DIR, 'dataset_summary.csv'), index=False)
    return summary


def fps_coverage_report(files, fps_list=(5, 10, 20, 30), max_files=20):
    """[R2-4] Justification of the frame rate. pretty_midi fills a frame only if
    floor(end*fs) > floor(start*fs); notes shorter than one frame can disappear.
    Reports, for each fs, the share of original notes that survive the piano-roll
    conversion and the mean onset shift caused by flooring to the frame grid."""
    note_arrays = []
    for f in files[:max_files]:
        try:
            pm_ = pretty_midi.PrettyMIDI(f)
        except Exception:
            continue
        notes = [n for inst in pm_.instruments for n in inst.notes]
        note_arrays.append((np.array([n.start for n in notes]), np.array([n.end for n in notes])))
    rows = []
    for fps in fps_list:
        kept = total = 0
        shift = []
        for starts, ends in note_arrays:
            kept += int(np.sum(np.floor(ends * fps) > np.floor(starts * fps)))
            total += len(starts)
            shift.append((starts * fps - np.floor(starts * fps)) / fps * 1000.0)
        rows.append({'fs': fps,
                     'frame_ms': round(1000.0 / fps, 1),
                     'context_window_s (50 steps)': round(seq_len / fps, 1),
                     'notes_surviving_pct': round(100.0 * kept / max(total, 1), 2),
                     'mean_onset_shift_ms': round(float(np.concatenate(shift).mean()), 1),
                     'n_notes_checked': total})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB_DIR, 'fps_coverage.csv'), index=False)
    return df


if RUN_DATASET_REPORT:
    DATASET_SUMMARY = dataset_report({'train': train_midi, 'validation': val_midi, 'test': test_midi})
    print(DATASET_SUMMARY.to_string(index=False))
if RUN_FPS_REPORT:
    print(fps_coverage_report(train_midi).to_string(index=False))


# ----- tokenizer: fitted on TRAIN pieces only -------------------------------
batch = 1
start_index = 0
note_tokenizer = NoteTokenizer()

for i in tqdm_notebook(range(len(sampled_200_midi))):
    dict_time_notes = generate_dict_time_notes(sampled_200_midi, batch_song=1, start_index=i, use_tqdm=False, fs=FRAME_PER_SECOND)
    full_notes = process_notes_in_song(dict_time_notes)
    for note in full_notes:
        note_tokenizer.partial_fit(list(note.values()))

note_tokenizer.add_new_note('e') 

unique_notes = note_tokenizer.unique_word
print(unique_notes)

TOTAL_SONGS = len(sampled_200_midi)


def tokenizer_report(tokenizer):
    """[R2-4] How polyphony is encoded and how large / sparse the vocabulary is."""
    keys = list(tokenizer.notes_freq.keys())
    freq = np.array([tokenizer.notes_freq[k] for k in keys], dtype=float)
    size = np.array([len(k.split(',')) for k in keys])
    rep = {'vocab_size_incl_rest_token': int(tokenizer.unique_word),
           'n_training_tokens (onset frames)': int(freq.sum()),
           'singleton_tokens_pct': round(100.0 * float((freq == 1).mean()), 2),
           'single_note_token_share_pct': round(100.0 * float(freq[size == 1].sum() / freq.sum()), 2),
           'chord_token_share_pct': round(100.0 * float(freq[size > 1].sum() / freq.sum()), 2),
           'mean_notes_per_token': round(float((size * freq).sum() / freq.sum()), 3),
           'max_notes_per_token': int(size.max())}
    return rep


def build_dataset(files, tokenizer, fs, seq_length, chunk=BATCH_SONG):
    """[R1-2][R2-3] Builds (X, y) arrays for a list of pieces (used for val / test)."""
    X_all, y_all, valid_all = [], [], []
    tokenizer.reset_oov()
    for s in range(0, len(files), chunk):
        n_chunk = chunk if len(files) >= chunk else len(files)
        inp, tgt = generate_batch_song(files, n_chunk, start_index=s, fs=fs, seq_len=seq_length, use_tqdm=False)
        if len(inp) == 0:
            continue
        # [FIX] mask of targets that exist in the training vocabulary (see evaluation note below)
        valid_all.append(np.array([all(t in tokenizer.notes_to_index for t in row) for row in tgt]))
        X_all.append(tokenizer.transform(inp))
        y_all.append(tokenizer.transform(tgt))
    oov_rate = tokenizer.n_oov / max(tokenizer.n_tokens_seen, 1)
    return np.concatenate(X_all), np.concatenate(y_all), oov_rate, np.concatenate(valid_all)


def piece_token_stream(roll_bool, tokenizer):
    """Frame-by-frame token indices of one piece (rest frames -> 'e'); same encoding as the
    training sequences. Used by the Markov / unigram baselines."""
    active = np.where(roll_bool.any(axis=0))[0]
    if len(active) == 0:
        return []
    e_idx = tokenizer.notes_to_index['e']
    stream = []
    for t in range(active[0], active[-1] + 1):
        pitches = np.nonzero(roll_bool[:, t])[0]
        if len(pitches) == 0:
            stream.append(e_idx)
        else:
            stream.append(tokenizer.notes_to_index.get(','.join(str(p) for p in pitches), e_idx))
    return stream


TOKENIZER_REPORT = tokenizer_report(note_tokenizer)
print(TOKENIZER_REPORT)

X_val, y_val, OOV_VAL, VAL_VALID = build_dataset(val_midi, note_tokenizer, FRAME_PER_SECOND, seq_len)
X_test, y_test, OOV_TEST, TEST_VALID = build_dataset(test_midi, note_tokenizer, FRAME_PER_SECOND, seq_len)
N_VAL_ALL, N_TEST_ALL = len(X_val), len(X_test)
TARGET_OOV_VAL, TARGET_OOV_TEST = 1.0 - VAL_VALID.mean(), 1.0 - TEST_VALID.mean()
# [FIX] EVALUATION NOTE. A validation/test target whose chord never occurs in the training pieces
# used to be mapped to the rest token 'e'. With ~40% unseen chords this made 'e' the right answer for
# roughly half of all targets, so every model (and even "always predict e") scored ~0.52 top-1 and the
# validation loss mostly measured how well 'e' is predicted. Validation / test / baseline metrics are
# therefore computed ONLY on sequences whose target is a real in-vocabulary token. Inputs that contain
# unseen chords are still mapped to 'e' (context noise), which is reported as the OOV rate.
X_val, y_val = X_val[VAL_VALID], y_val[VAL_VALID]
X_test, y_test = X_test[TEST_VALID], y_test[TEST_VALID]
print("sequences kept (in-vocabulary targets): val {}/{} | test {}/{} | token OOV rate val {:.1%} test {:.1%}".format(
    len(X_val), N_VAL_ALL, len(X_test), N_TEST_ALL, OOV_VAL, OOV_TEST))


class SeqSelfAttention(tf.keras.layers.Layer):

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):

        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.attention_activation = tf.keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = tf.keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

        super(SeqSelfAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': tf.keras.regularizers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.regularizers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'attention_activation': tf.keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = input_shape[2]

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = input_shape[2]

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            inputs, positions = inputs
            positions = K.cast(positions, 'int32')
            mask = mask[1]
        else:
            positions = None

        input_len = K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.attention_width is not None:
            ones = tf.ones((input_len, input_len))
            if self.history_only:
                local = tf.linalg.band_part(
                    ones,
                    K.minimum(input_len, self.attention_width - 1),
                    0,
                )
            else:
                local = tf.linalg.band_part(
                    ones,
                    K.minimum(input_len, self.attention_width // 2),
                    K.minimum(input_len, (self.attention_width - 1) // 2),
                )
            e = e * K.expand_dims(local, 0)
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            e = K.permute_dimensions(K.permute_dimensions(e * mask, (0, 2, 1)) * mask, (0, 2, 1))

        # a_{t} = \text{softmax}(e_t)
        s = K.sum(e, axis=-1)
        s = K.tile(K.expand_dims(s, axis=-1), K.stack([1, 1, input_len]))
        a = e / (s + K.epsilon())

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if positions is not None:
            pos_num = K.shape(positions)[1]
            batch_indices = K.tile(K.expand_dims(K.arange(K.shape(inputs)[0]), axis=-1), K.stack([1, pos_num]))
            pos_indices = K.stack([batch_indices, positions], axis=-1)
            v = tf.gather_nd(v, pos_indices)
            a = tf.gather_nd(a, pos_indices)

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q, k = K.dot(inputs, self.Wt), K.dot(inputs, self.Wx)
        q = K.tile(K.expand_dims(q, 2), K.stack([1, 1, input_len, 1]))
        k = K.tile(K.expand_dims(k, 1), K.stack([1, input_len, 1, 1]))
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e = e + self.ba
        return e

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape, pos_shape = input_shape
            output_shape = (input_shape[0], pos_shape[1], input_shape[2])
        else:
            output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            mask = mask[1]
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - tf.eye(input_len))) / batch_size

    @staticmethod
    def get_custom_objects():
      return {'SeqSelfAttention': SeqSelfAttention}


# =============================================================================
# SECTION C -- MODEL + TRAINING WITH VALIDATION           [R1-2][R2-3][R2-5]
# =============================================================================
def create_model(seq_len, unique_notes, dropout=0.3, output_emb=100, rnn_unit=128, dense_unit=64,
                 attention_width=50, rnn_type='GRU'):
  # [R2-5] rnn_type='LSTM' builds the identical architecture with LSTM cells (baseline).
  RNN = tf.keras.layers.GRU if rnn_type.upper() == 'GRU' else tf.keras.layers.LSTM
  inputs = tf.keras.layers.Input(shape=(seq_len,))
  embedding = tf.keras.layers.Embedding(input_dim=unique_notes+1, output_dim=output_emb, input_length=seq_len)(inputs)
  forward_pass = tf.keras.layers.Bidirectional(RNN(rnn_unit, return_sequences=True))(embedding)
  forward_pass , att_vector = SeqSelfAttention(
      return_attention=True,
      attention_activation='sigmoid', 
      attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
      attention_width=attention_width, 
      kernel_regularizer=tf.keras.regularizers.l2(1e-4),
      bias_regularizer=tf.keras.regularizers.l1(1e-4),
      attention_regularizer_weight=1e-4,
  )(forward_pass)
  forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
  forward_pass = tf.keras.layers.Bidirectional(RNN(rnn_unit, return_sequences=True))(forward_pass)
  forward_pass , att_vector2 = SeqSelfAttention(
      return_attention=True,
      attention_activation='sigmoid', 
      attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
      attention_width=attention_width, 
      kernel_regularizer=tf.keras.regularizers.l2(1e-4),
      bias_regularizer=tf.keras.regularizers.l1(1e-4),
      attention_regularizer_weight=1e-4,
  )(forward_pass)
  forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
  forward_pass = tf.keras.layers.Bidirectional(RNN(rnn_unit))(forward_pass)
  forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
  forward_pass = tf.keras.layers.Dense(dense_unit)(forward_pass)
  forward_pass = tf.keras.layers.LeakyReLU()(forward_pass)
  outputs = tf.keras.layers.Dense(unique_notes+1, activation = "softmax")(forward_pass)

  model = tf.keras.Model(inputs=inputs, outputs=outputs, name='generate_scores_' + rnn_type.lower())
  return model


def evaluate_model(mdl, X, y, batch=256):
    """[R1-2][R2-1][R2-5] Next-token loss (nats), perplexity and top-1 accuracy on (X, y)."""
    total_loss, total_correct, n = 0.0, 0, 0
    for s in range(0, len(X), batch):
        xb, yb = X[s:s + batch], y[s:s + batch]
        preds = mdl(xb, training=False)
        total_loss += float(tf.reduce_sum(sparse_categorical_crossentropy(yb, preds)))
        total_correct += int(np.sum(np.argmax(preds.numpy(), axis=1) == yb[:, 0]))
        n += len(xb)
    loss = total_loss / max(n, 1)
    return {'loss': loss, 'perplexity': float(np.exp(loss)), 'top1_acc': total_correct / max(n, 1), 'n_sequences': n}


class TrainModel:
  
  def __init__(self, epochs, note_tokenizer, sampled_200_midi, frame_per_second, 
               batch_nnet_size, batch_song, optimizer, checkpoint, loss_fn,
               checkpoint_prefix, total_songs, model, val_data=None, name='gru'):
    self.epochs = epochs
    self.note_tokenizer = note_tokenizer
    self.sampled_200_midi = sampled_200_midi
    self.frame_per_second = frame_per_second
    self.batch_nnet_size = batch_nnet_size
    self.batch_song = batch_song
    self.optimizer = optimizer
    self.checkpoint = checkpoint
    self.loss_fn = loss_fn
    self.checkpoint_prefix = checkpoint_prefix
    self.total_songs = total_songs
    self.model = model
    self.val_data = val_data            # [R1-2][R2-3] (X_val, y_val)
    self.name = name
    self.history = []                   # [R1-2] one dict per epoch -> loss curves
    self.best_epoch = None
    self.train_seconds = None
    
  def train(self):
    t_start = time.time()
    best_val, best_weights = np.inf, None
    for epoch in tqdm_notebook(range(self.epochs),desc='epochs'):

      shuffle(self.sampled_200_midi)
      loss_total = 0.0
      acc_total = 0.0
      n_seen = 0
      steps = 0
      steps_nnet = 0

      for i in tqdm_notebook(range(0,self.total_songs, self.batch_song), desc='MUSIC'):

        steps += 1
        inputs_nnet_large, outputs_nnet_large = generate_batch_song(
            self.sampled_200_midi, self.batch_song, start_index=i, fs=self.frame_per_second, 
            seq_len=seq_len, use_tqdm=False)
        inputs_nnet_large = np.array(self.note_tokenizer.transform(inputs_nnet_large), dtype=np.int32)
        outputs_nnet_large = np.array(self.note_tokenizer.transform(outputs_nnet_large), dtype=np.int32)

        index_shuffled = np.arange(start=0, stop=len(inputs_nnet_large))
        np.random.shuffle(index_shuffled)

        for nnet_steps in tqdm_notebook(range(0,len(index_shuffled),self.batch_nnet_size)):
          steps_nnet += 1
          current_index = index_shuffled[nnet_steps:nnet_steps+self.batch_nnet_size]
          inputs_nnet, outputs_nnet = inputs_nnet_large[current_index], outputs_nnet_large[current_index]
          
          if len(inputs_nnet) // self.batch_nnet_size != 1:
            break
          loss, acc = self.train_step(inputs_nnet, outputs_nnet)
          loss_total += float(tf.math.reduce_sum(loss))
          acc_total += float(acc) * len(inputs_nnet)
          n_seen += len(inputs_nnet)
          if steps_nnet % 20 == 0:
            print("[{}] epochs {} | Steps {} | mean train loss : {:.4f}".format(
                self.name, epoch + 1, steps_nnet, loss_total / max(n_seen, 1)))

      # [R1-2][R2-3] per-epoch bookkeeping with validation
      rec = {'epoch': epoch + 1,
             'train_loss': loss_total / max(n_seen, 1),
             'train_top1_acc': acc_total / max(n_seen, 1),
             'n_train_sequences_seen': n_seen}
      if self.val_data is not None:
        v = evaluate_model(self.model, *self.val_data)
        rec.update({'val_loss': v['loss'], 'val_perplexity': v['perplexity'], 'val_top1_acc': v['top1_acc']})
        if v['loss'] < best_val:
          best_val, best_weights, self.best_epoch = v['loss'], self.model.get_weights(), epoch + 1
      self.history.append(rec)
      print("[{}] epoch {} summary: {}".format(self.name, epoch + 1, {k: round(float(x), 4) for k, x in rec.items()}))

      self.checkpoint.save(file_prefix = self.checkpoint_prefix)

    if best_weights is not None:
      self.model.set_weights(best_weights)       # keep the epoch with the lowest validation loss
    self.train_seconds = time.time() - t_start
    pd.DataFrame(self.history).to_csv(os.path.join(TAB_DIR, 'training_history_{}.csv'.format(self.name)), index=False)
  
  @tf.function
  def train_step(self, inputs, targets):
    with tf.GradientTape() as tape:
      prediction = self.model(inputs, training=True)
      loss = self.loss_fn(targets, prediction)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=-1, output_type=tf.int32),
                                          tf.reshape(targets, [-1])), tf.float32))
    return loss, acc


def train_one_model(rnn_type='GRU'):
    """Builds, trains (with validation) and returns (model, trainer)."""
    mdl = create_model(seq_len, unique_notes, dropout=DROPOUT, output_emb=OUTPUT_EMB, rnn_unit=RNN_UNIT,
                       dense_unit=DENSE_UNIT, attention_width=ATTENTION_WIDTH, rnn_type=rnn_type)
    summary_lines = []
    mdl.summary(print_fn=lambda s: summary_lines.append(s))
    print("\n".join(summary_lines))
    with open(os.path.join(TAB_DIR, 'model_summary_{}.txt'.format(rnn_type.lower())), 'w') as fh:
        fh.write("\n".join(summary_lines))
    try:
        tf.keras.utils.plot_model(mdl, os.path.join(FIG_DIR, 'model_{}.png'.format(rnn_type.lower())), show_shapes=True)
    except Exception as e:        # graphviz / pydot may be missing
        print("plot_model skipped:", e)

    opt = Nadam()
    ckpt = tf.train.Checkpoint(optimizer=opt, model=mdl)
    ckpt_dir = os.path.join('./training_checkpoints', rnn_type.lower())
    ckpt_prefix = os.path.join(ckpt_dir, "ckpt")
    trainer = TrainModel(EPOCHS, note_tokenizer, sampled_200_midi, FRAME_PER_SECOND,
                         BATCH_NNET_SIZE, BATCH_SONG, opt, ckpt, sparse_categorical_crossentropy,
                         ckpt_prefix, TOTAL_SONGS, mdl, val_data=(X_val, y_val), name=rnn_type.lower())
    trainer.train()
    return mdl, trainer


model, train_class = train_one_model('GRU')

MODEL_PATH = 'model_ep{}.h5'.format(EPOCHS)          # name now matches the real epoch count
model.save(MODEL_PATH)
pickle.dump( note_tokenizer, open( "tokenizer.p", "wb" ) )

model = tf.keras.models.load_model(MODEL_PATH, custom_objects=SeqSelfAttention.get_custom_objects())
note_tokenizer  = pickle.load( open( "tokenizer.p", "rb" ) )

lstm_model, lstm_trainer = (None, None)
if RUN_LSTM_BASELINE:                                 # [R2-5]
    lstm_model, lstm_trainer = train_one_model('LSTM')
    lstm_model.save('model_lstm_ep{}.h5'.format(EPOCHS))


def plot_training_curves(trainer, path):
    """[R1-2] One figure PER MODEL: loss and next-token accuracy per epoch, train vs validation.
    The dotted vertical line marks the epoch whose weights were kept (lowest validation loss)."""
    h = pd.DataFrame(trainer.history)
    name = trainer.name.upper()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h['epoch'], h['train_loss'], marker='o', label='train')
    if 'val_loss' in h:
        axes[0].plot(h['epoch'], h['val_loss'], marker='s', linestyle='--', label='validation')
        axes[1].plot(h['epoch'], h['val_top1_acc'], marker='s', linestyle='--', label='validation')
    axes[1].plot(h['epoch'], h['train_top1_acc'], marker='o', label='train')
    for ax in axes:
        if trainer.best_epoch is not None:
            ax.axvline(trainer.best_epoch, color='grey', linestyle=':', label='best val. epoch ({})'.format(trainer.best_epoch))
        ax.set_xlabel('Epoch'); ax.grid(alpha=.3); ax.legend()
    axes[0].set_ylabel('Cross-entropy loss (nats)'); axes[0].set_title('{} - (a) Loss'.format(name))
    axes[1].set_ylabel('Top-1 accuracy'); axes[1].set_title('{} - (b) Next-token accuracy'.format(name))
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.show()


plot_training_curves(train_class, os.path.join(FIG_DIR, 'training_curves_gru.png'))
if lstm_trainer is not None:
    plot_training_curves(lstm_trainer, os.path.join(FIG_DIR, 'training_curves_lstm.png'))


def export_hyperparameter_table(mdl, trainer, rnn_type='GRU'):
    """[R1-2][R2-3] The 'clear table' with architecture, training set-up, libraries and hardware."""
    try:
        lr = float(tf.keras.backend.get_value(trainer.optimizer.learning_rate))
    except Exception:
        lr = 'default Nadam (0.002)'
    gpus = tf.config.list_physical_devices('GPU')
    try:
        gpu_names = [tf.config.experimental.get_device_details(g).get('device_name', 'GPU') for g in gpus]
    except Exception:
        gpu_names = [str(g) for g in gpus]
    rows = [
        ('Data', 'Dataset / version', os.path.basename(MAESTRO_DIR)),
        ('Data', 'MIDI files found in folder', len(list_all_midi)),
        ('Data', 'Split protocol', SPLIT_SOURCE),
        ('Data', 'Pieces train / val / test', '{} / {} / {}'.format(len(train_midi), len(val_midi), len(test_midi))),
        ('Data', 'Sequences val / test (in-vocabulary targets only)', '{} / {}'.format(len(X_val), len(X_test))),
        ('Data', 'Sequences val / test before removing OOV targets', '{} / {}'.format(N_VAL_ALL, N_TEST_ALL)),
        ('Data', 'Share of targets removed (OOV) val / test', '{:.1%} / {:.1%}'.format(TARGET_OOV_VAL, TARGET_OOV_TEST)),
        ('Data', 'Training sequences per epoch (full batches)', trainer.history[-1]['n_train_sequences_seen'] if trainer.history else 'n/a'),
        ('Data', 'OOV token rate val / test', '{:.1%} / {:.1%}'.format(OOV_VAL, OOV_TEST)),
        ('Data', 'Frames per second (fs)', FRAME_PER_SECOND),
        ('Data', 'Input sequence length (steps)', seq_len),
        ('Data', 'Context window (s)', seq_len / FRAME_PER_SECOND),
        ('Data', 'Vocabulary size (incl. rest token)', unique_notes),
        ('Model', 'Cell type', rnn_type),
        ('Model', 'Embedding dimension', OUTPUT_EMB),
        ('Model', 'Recurrent layers', '3 x Bidirectional({}) '.format(rnn_type)),
        ('Model', 'Units per direction (layer output = 2x)', RNN_UNIT),
        ('Model', 'Self-attention layers', '2 (after recurrent layer 1 and 2)'),
        ('Model', 'Self-attention type / heads / width', 'SeqSelfAttention multiplicative / {} head / {}'.format(ATTENTION_HEADS, ATTENTION_WIDTH)),
        ('Model', 'Attention activation / regularisers', 'sigmoid / L2 1e-4 (kernel), L1 1e-4 (bias), attention reg. 1e-4'),
        ('Model', 'Dropout rate (after att.1, att.2, last recurrent layer)', DROPOUT),
        ('Model', 'Dense layer', '{} units + LeakyReLU'.format(DENSE_UNIT)),
        ('Model', 'Output dimension', '{} (softmax)'.format(unique_notes + 1)),
        ('Model', 'Trainable parameters', int(mdl.count_params())),
        ('Training', 'Optimizer / learning rate', 'Nadam / {}'.format(lr)),
        ('Training', 'Loss', 'sparse categorical cross-entropy'),
        ('Training', 'Gradient-step batch size (sequences)', BATCH_NNET_SIZE),
        ('Training', 'Pieces decoded per loading chunk', BATCH_SONG),
        ('Training', 'Epochs', EPOCHS),
        ('Training', 'Epoch with best validation loss (weights kept)', trainer.best_epoch),
        ('Training', 'Training time (s)', round(trainer.train_seconds or 0, 1)),
        ('Training', 'Random seed', SEED),
        ('Generation', 'Temperature', TEMPERATURE),
        ('Generation', 'Generated notes per file / runs per condition', '{} / {}'.format(MAX_GENERATE, N_RUNS)),
        ('Generation', 'Playback rate (fs)', PLAYBACK_FS),
        ('Software', 'Python', sys.version.split()[0]),
        ('Software', 'TensorFlow / Keras', '{} / {}'.format(tf.__version__, getattr(tf.keras, '__version__', 'n/a'))),
        ('Software', 'NumPy', np.__version__),
        ('Software', 'pretty_midi', getattr(pretty_midi, '__version__', 'n/a')),
        ('Software', 'pandas / matplotlib', '{} / {}'.format(pd.__version__, __import__('matplotlib').__version__)),
        ('Hardware', 'GPU', ', '.join(gpu_names) if gpu_names else 'none (CPU)'),
        ('Hardware', 'CPU / OS', '{} / {}'.format(platform.processor() or platform.machine(), platform.platform())),
    ]
    df = pd.DataFrame(rows, columns=['Group', 'Parameter', 'Value'])
    df.to_csv(os.path.join(TAB_DIR, 'hyperparameters_{}.csv'.format(rnn_type.lower())), index=False)
    with open(os.path.join(TAB_DIR, 'tokenizer_report.json'), 'w') as fh:
        json.dump(TOKENIZER_REPORT, fh, indent=2)
    print(df.to_string(index=False))
    return df


HYPERPARAMS_GRU = export_hyperparameter_table(model, train_class, 'GRU')
if lstm_model is not None:
    HYPERPARAMS_LSTM = export_hyperparameter_table(lstm_model, lstm_trainer, 'LSTM')



# =============================================================================
# SECTION D -- BASELINES                                  [R2-5]
# =============================================================================
class MarkovBaseline:
    """[R2-5] Order-1 Markov chain (order=0 -> unigram) over the SAME frame tokens the
    networks use. Smoothing: p(next|prev) = (count(prev,next) + beta*p_uni(next)) / (count(prev)+beta)."""

    def __init__(self, streams, vocab_size, order=1, beta=1.0):
        self.order, self.beta, self.V = order, beta, vocab_size
        self.uni = Counter()
        self.trans = defaultdict(Counter)
        for s in streams:
            self.uni.update(s)
            if order >= 1:
                for a, b in zip(s[:-1], s[1:]):
                    self.trans[a][b] += 1
        self.N = sum(self.uni.values())
        self.tot = {k: sum(v.values()) for k, v in self.trans.items()}
        self.best = {k: v.most_common(1)[0][0] for k, v in self.trans.items()}
        self.best_uni = self.uni.most_common(1)[0][0]
        self._uni_tokens = np.array(list(self.uni.keys()))
        p = np.array(list(self.uni.values()), dtype=float)
        self._uni_p = p / p.sum()
        self._cache = {}

    def p_uni(self, tok):
        return (self.uni.get(tok, 0) + 0.5) / (self.N + 0.5 * self.V)

    def p_next(self, prev, nxt):
        if self.order == 0 or prev not in self.trans:
            return self.p_uni(nxt)
        return (self.trans[prev].get(nxt, 0) + self.beta * self.p_uni(nxt)) / (self.tot[prev] + self.beta)

    def evaluate(self, X, y):
        """Same (X, y) pairs as the networks: prev = last input token, target = y."""
        prev, tgt = X[:, -1], y[:, 0]
        nll = -np.mean([np.log(self.p_next(int(a), int(b))) for a, b in zip(prev, tgt)])
        if self.order == 0:
            acc = float(np.mean(tgt == self.best_uni))
        else:
            acc = float(np.mean([self.best.get(int(a), self.best_uni) == int(b) for a, b in zip(prev, tgt)]))
        return {'loss': float(nll), 'perplexity': float(np.exp(nll)), 'top1_acc': acc, 'n_sequences': len(tgt)}

    def _dist(self, prev):
        if prev not in self._cache:
            toks = np.array(list(self.trans[prev].keys()))
            p = np.array(list(self.trans[prev].values()), dtype=float)
            self._cache[prev] = (toks, p / p.sum())
        return self._cache[prev]

    def generate(self, n, rng=np.random):
        seq = [int(rng.choice(self._uni_tokens, p=self._uni_p))]
        while len(seq) < n:
            prev = seq[-1]
            if self.order == 0 or prev not in self.trans:
                nxt = rng.choice(self._uni_tokens, p=self._uni_p)
            else:
                toks, p = self._dist(prev)
                nxt = rng.choice(toks, p=p)
            seq.append(int(nxt))
        return seq


train_streams = [piece_token_stream(get_bool_roll(f, FRAME_PER_SECOND), note_tokenizer) for f in train_midi]
unigram_baseline = MarkovBaseline(train_streams, unique_notes + 1, order=0)
markov_baseline = MarkovBaseline(train_streams, unique_notes + 1, order=1)

# [R2-1][R2-5] next-token comparison on the held-out TEST pieces (identical pairs for all models)
rows = [dict(model='BiGRU + self-attention (proposed)', params=int(model.count_params()), **evaluate_model(model, X_test, y_test))]
if lstm_model is not None:
    rows.append(dict(model='BiLSTM + self-attention', params=int(lstm_model.count_params()), **evaluate_model(lstm_model, X_test, y_test)))
rows.append(dict(model='Markov chain (order 1)', params=len(markov_baseline.trans), **markov_baseline.evaluate(X_test, y_test)))
rows.append(dict(model='Unigram (token frequency)', params=len(unigram_baseline.uni), **unigram_baseline.evaluate(X_test, y_test)))
MODEL_COMPARISON = pd.DataFrame(rows)
MODEL_COMPARISON.to_csv(os.path.join(TAB_DIR, 'model_comparison_test.csv'), index=False)
print(MODEL_COMPARISON.round(4).to_string(index=False))


# =============================================================================
# SECTION E -- GENERATION                                 [R1-5][R2-7][FIX]
# =============================================================================
def generate_from_random(unique_notes, seq_len=50, rng=np.random):
  # [FIX] token ids run from 1..unique_notes; the original randint(0, unique_notes) could
  # draw id 0 (not in the vocabulary -> KeyError when writing MIDI) and never drew 'e'.
  generate = rng.randint(1, unique_notes + 1, seq_len).tolist()
  return generate

def generate_from_unigram_seed(note_tokenizer, seq_len=50, rng=np.random):
  # [R1-5] extra control: a seed window sampled from the training token distribution
  # instead of uniformly over the vocabulary (uniform chords are mostly rare, unmusical tokens).
  keys = list(note_tokenizer.notes_freq.keys())
  p = np.array([note_tokenizer.notes_freq[k] for k in keys], dtype=float)
  idx = rng.choice(len(keys), seq_len, p=p / p.sum())
  return [note_tokenizer.notes_to_index[keys[i]] for i in idx]
    
def generate_from_one_note(note_tokenizer, new_notes='35'):
  if new_notes not in note_tokenizer.notes_to_index:
    raise ValueError("start token '{}' is not a single-note token of the vocabulary".format(new_notes))
  generate = [note_tokenizer.notes_to_index['e'] for i in range(49)]
  generate += [note_tokenizer.notes_to_index[new_notes]]
  return generate

def generate_notes(generate, model, unique_notes, max_generated=1000, seq_len=50, temperature=1.0, rng=np.random):
  generate = list(generate)
  for i in tqdm_notebook(range(max_generated), desc='genrt', leave=False):
    test_input = np.array([generate])[:,i:i+seq_len]
    probs = model(test_input, training=False).numpy()[0].astype(np.float64)
    probs[0] = 0.0                                   # [FIX] id 0 is not a real token
    if temperature != 1.0:                           # [R1-5] explicit sampling temperature
      probs = np.power(np.maximum(probs, 1e-12), 1.0 / temperature)
    probs = probs / probs.sum()                      # [FIX] float32 softmax may not sum to 1 for choice()
    generate.append(int(rng.choice(unique_notes + 1, p=probs)))
  return generate


def generated_to_roll(generate, tokenizer, start_index=49, max_generated=1000):
  note_string = [tokenizer.index_to_notes[int(ind_note)] for ind_note in generate]
  array_piano_roll = np.zeros((128, max_generated+1), dtype=np.int16)
  n_rest = 0
  for index, note in enumerate(note_string[start_index:]):
    if note == 'e':
      n_rest += 1
    else:
      for j in note.split(','):
        array_piano_roll[int(j), index] = 1
  return array_piano_roll, n_rest / max(len(note_string[start_index:]), 1)


def write_midi_file_from_generated(generate, midi_file_name = "result.mid", start_index=49, fs=8, max_generated=1000):
  array_piano_roll, rest_ratio = generated_to_roll(generate, note_tokenizer, start_index, max_generated)
  generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
  try:
    print("Tempo {}".format(generate_to_midi.estimate_tempo()))
  except Exception:
    pass
  for note in generate_to_midi.instruments[0].notes:
    note.velocity = 100
  generate_to_midi.write(midi_file_name)
  return generate_to_midi, rest_ratio


# =============================================================================
# SECTION F -- OBJECTIVE METRICS                          [R1-1][R2-1a][R2-7]
# Each metric is tied to one claim of the manuscript:
#   "more concentrated pitch distribution" -> pitch_range, pitch_std, pc_entropy_bits, pitch_entropy_bits
#   "relatively regular temporal structure" -> ioi_cv, ioi_entropy_bits, ngram_repetition
#   "broader, less uniform distribution"    -> pitch_entropy_bits, early_density_ratio
#   "slower tempo"                          -> onset_density_per_s, ioi_mean_s, rest_ratio, tempo_est_bpm
#   "variety of note durations"             -> dur_unique, dur_entropy_bits
#   "resembles real music"                  -> js_pc_to_train, js_pitch_to_train
# =============================================================================
def _entropy(counts, base=2):
    counts = np.asarray(counts, dtype=float)
    s = counts.sum()
    if s <= 0:
        return float('nan')
    p = counts[counts > 0] / s
    return float(-(p * np.log(p)).sum() / np.log(base))

def pitch_class_hist(pitches):
    return np.bincount(np.asarray(pitches, dtype=int) % 12, minlength=12).astype(float)

def js_divergence(p, q):
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint)."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def reference_histograms(files, fs):
    """Pitch / pitch-class histogram of NOTE ONSETS in real pieces (same counting rule as compute_metrics)."""
    pitch_hist = np.zeros(128)
    for f in files:
        r = get_bool_roll(f, fs).astype(np.int8)
        onsets = np.diff(np.pad(r, [(0, 0), (1, 0)]), axis=1) == 1
        pitch_hist += onsets.sum(axis=1)
    pc_hist = np.bincount(np.arange(128) % 12, weights=pitch_hist, minlength=12)
    return pc_hist, pitch_hist


def compute_metrics(pm, ref_pc_hist=None, ref_pitch_hist=None, ngram=4):
    """[R1-1][R2-1a] Objective descriptors of one MIDI object. Returns None for (almost) empty files."""
    notes = sorted((n for inst in pm.instruments for n in inst.notes), key=lambda n: (n.start, n.pitch))
    if len(notes) < ngram + 1:
        return None
    pitches = np.array([n.pitch for n in notes], dtype=int)
    starts = np.array([n.start for n in notes])
    ends = np.array([n.end for n in notes])
    durs = ends - starts
    total_time = float(ends.max())

    groups = defaultdict(list)                       # simultaneous onsets -> chord
    for n in notes:
        groups[round(n.start, 4)].append(n.pitch)
    onset_times = np.array(sorted(groups))
    ioi = np.diff(onset_times)

    pc_hist = pitch_class_hist(pitches)
    pitch_hist = np.bincount(pitches, minlength=128).astype(float)
    dur_r = np.round(durs, 3)
    ioi_r = np.round(ioi, 3)

    seq = [tuple(sorted(groups[k])) for k in onset_times]
    grams = [tuple(seq[i:i + ngram]) for i in range(len(seq) - ngram + 1)]
    repetition = 1.0 - len(set(grams)) / len(grams) if grams else float('nan')   # share of repeated 4-grams

    q = total_time / 4.0                              # [R2-7] "clustering at the beginning"
    early = np.sum(starts < q) / q
    late = np.sum(starts >= q) / max(total_time - q, 1e-9)

    try:
        tempo = float(pm.estimate_tempo())
    except Exception:
        tempo = float('nan')

    m = {
        'n_notes': len(notes),
        'duration_s': total_time,
        'note_density_per_s': len(notes) / total_time,
        'onset_density_per_s': len(onset_times) / total_time,
        'mean_polyphony': float(np.mean([len(v) for v in groups.values()])),
        'pitch_min': int(pitches.min()), 'pitch_max': int(pitches.max()),
        'pitch_range': int(pitches.max() - pitches.min()),
        'pitch_mean': float(pitches.mean()), 'pitch_std': float(pitches.std()),
        'pitch_entropy_bits': _entropy(pitch_hist),
        'pc_entropy_bits': _entropy(pc_hist),
        'n_pitch_classes_used': int((pc_hist > 0).sum()),
        'dur_mean_s': float(durs.mean()), 'dur_std_s': float(durs.std()),
        'dur_unique': int(len(set(dur_r))),
        'dur_entropy_bits': _entropy(list(Counter(dur_r).values())),
        'ioi_mean_s': float(ioi.mean()) if len(ioi) else float('nan'),
        'ioi_cv': float(ioi.std() / ioi.mean()) if len(ioi) and ioi.mean() > 0 else float('nan'),
        'ioi_entropy_bits': _entropy(list(Counter(ioi_r).values())) if len(ioi) else float('nan'),
        'ngram_repetition': repetition,
        'early_density_ratio': float(early / late) if late > 0 else float('nan'),
        'tempo_est_bpm': tempo,
    }
    if ref_pc_hist is not None:
        m['js_pc_to_train'] = js_divergence(pc_hist, ref_pc_hist)
    if ref_pitch_hist is not None:
        m['js_pitch_to_train'] = js_divergence(pitch_hist, ref_pitch_hist)
    m['_pc_hist'], m['_pitch_hist'], m['_durs'] = pc_hist, pitch_hist, durs
    return m


def holm_adjust(pvals):
    """Holm-Bonferroni adjusted p-values (family = all metrics of one comparison)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adj[idx] = running
    return adj


def compare_conditions(df, cond_a, cond_b, metric_names):
    """[R1-1][R2-7] Two-sided Mann-Whitney U test between two generation conditions."""
    rows = []
    for name in metric_names:
        xa = df.loc[df['condition'] == cond_a, name].dropna()
        xb = df.loc[df['condition'] == cond_b, name].dropna()
        if len(xa) < 2 or len(xb) < 2:
            continue
        row = {'metric': name, cond_a + '_mean': xa.mean(), cond_b + '_mean': xb.mean()}
        if scipy_stats is not None:
            u, p = scipy_stats.mannwhitneyu(xa, xb, alternative='two-sided')
            row.update({'U': u, 'p_value': p})
        rows.append(row)
    out = pd.DataFrame(rows)
    if 'p_value' in out and len(out):
        out['p_holm'] = holm_adjust(out['p_value'].values)    # [R1-1] 16 metrics -> correct for multiple tests
    return out



# =============================================================================
# SECTION E (cont.) -- FIXED EXPERIMENTAL CONDITIONS      [R1-5][R2-5][R2-7]
# Every condition: N_RUNS independent generations, seeds SEED+0..SEED+N_RUNS-1,
# same length (MAX_GENERATE), same playback rate, same temperature.
# =============================================================================
unique_notes = note_tokenizer.unique_word
GEN_CONDITIONS = {}

def register_neural_conditions(prefix, mdl, with_unigram_seed=False):
    GEN_CONDITIONS[prefix + '_random_uniform'] = lambda rng: generate_notes(
        generate_from_random(unique_notes, seq_len, rng), mdl, unique_notes, MAX_GENERATE, seq_len, TEMPERATURE, rng)
    if with_unigram_seed:
        GEN_CONDITIONS[prefix + '_random_unigram'] = lambda rng: generate_notes(
            generate_from_unigram_seed(note_tokenizer, seq_len, rng), mdl, unique_notes, MAX_GENERATE, seq_len, TEMPERATURE, rng)
    GEN_CONDITIONS[prefix + '_one_note'] = lambda rng: generate_notes(
        generate_from_one_note(note_tokenizer, ONE_NOTE_START), mdl, unique_notes, MAX_GENERATE, seq_len, TEMPERATURE, rng)

register_neural_conditions('gru', model, with_unigram_seed=True)
if lstm_model is not None:
    register_neural_conditions('lstm', lstm_model)
GEN_CONDITIONS['baseline_unigram'] = lambda rng: unigram_baseline.generate(seq_len + MAX_GENERATE, rng)
GEN_CONDITIONS['baseline_markov1'] = lambda rng: markov_baseline.generate(seq_len + MAX_GENERATE, rng)

CONDITION_DESCRIPTIONS = {
    'gru_random_uniform': 'GRU model; seed window = 50 token ids drawn uniformly from the vocabulary (original "random notes")',
    'gru_random_unigram': 'GRU model; seed window = 50 tokens drawn from the training token distribution (control)',
    'gru_one_note': "GRU model; seed window = 49 rest tokens + single note '{}' (original 'single note')".format(ONE_NOTE_START),
    'lstm_random_uniform': 'LSTM baseline; same as gru_random_uniform',
    'lstm_one_note': 'LSTM baseline; same as gru_one_note',
    'baseline_unigram': 'i.i.d. sampling from the training token distribution',
    'baseline_markov1': 'order-1 Markov chain over frame tokens, trained on the same pieces',
}
pd.DataFrame([{'condition': c, 'description': CONDITION_DESCRIPTIONS.get(c, ''), 'n_runs': N_RUNS,
               'seeds': 'SEED+0..SEED+{} (SEED={})'.format(N_RUNS - 1, SEED),
               'generated_tokens_per_file': MAX_GENERATE, 'temperature': TEMPERATURE,
               'playback_fs': PLAYBACK_FS} for c in GEN_CONDITIONS]
             ).to_csv(os.path.join(TAB_DIR, 'generation_conditions.csv'), index=False)

REF_PC_HIST, REF_PITCH_HIST = reference_histograms(train_midi, FRAME_PER_SECOND)

metric_rows = []
pool = defaultdict(lambda: {'pc': np.zeros(12), 'pitch': np.zeros(128), 'durs': []})
clips = defaultdict(list)                       # pretty_midi objects, used for figures / listening test

def _record(cond, run, pm_obj, rest_ratio):
    m = compute_metrics(pm_obj, REF_PC_HIST, REF_PITCH_HIST)
    if m is None:
        print("skipped empty file:", cond, run)
        return
    pool[cond]['pc'] += m.pop('_pc_hist'); pool[cond]['pitch'] += m.pop('_pitch_hist'); pool[cond]['durs'].append(m.pop('_durs'))
    m.update({'condition': cond, 'run': run, 'rest_ratio': rest_ratio})
    metric_rows.append(m)
    clips[cond].append(pm_obj)

for cond, gen_fn in GEN_CONDITIONS.items():
    os.makedirs(os.path.join(MIDI_DIR, cond), exist_ok=True)
    for r in range(N_RUNS):
        rng = np.random.RandomState(SEED + r)
        tokens = gen_fn(rng)
        pm_obj, rest_ratio = write_midi_file_from_generated(
            tokens, os.path.join(MIDI_DIR, cond, 'run_{:02d}.mid'.format(r)),
            start_index=seq_len - 1, fs=PLAYBACK_FS, max_generated=MAX_GENERATE)
        if r == 0 and cond == 'gru_random_uniform':      # keep the original file names
            pm_obj.write('random.mid')
        if r == 0 and cond == 'gru_one_note':
            pm_obj.write('one_note.mid')
        _record(cond, r, pm_obj, rest_ratio)

# real music reference: equal-length excerpts of held-out test pieces, same fs and same piano-roll pipeline
rng_real = np.random.RandomState(SEED)
for r in range(N_RUNS):
    f = test_midi[r % len(test_midi)]
    roll = get_bool_roll(f, FRAME_PER_SECOND)
    n_frames = MAX_GENERATE + 1
    active = np.where(roll.any(axis=0))[0]
    lo_t = active[0]; hi_t = max(active[-1] - n_frames, lo_t)
    s = int(rng_real.randint(lo_t, hi_t + 1))
    excerpt = roll[:, s:s + n_frames].astype(np.int16)
    if excerpt.shape[1] < n_frames:
        excerpt = np.pad(excerpt, [(0, 0), (0, n_frames - excerpt.shape[1])])
    real_pm = piano_roll_to_pretty_midi(excerpt, fs=PLAYBACK_FS)
    for note in real_pm.instruments[0].notes:
        note.velocity = 100
    _record('real_test_excerpt', r, real_pm, float((~excerpt.any(axis=0)).mean()))

METRICS_PER_RUN = pd.DataFrame(metric_rows)
METRICS_PER_RUN.to_csv(os.path.join(TAB_DIR, 'metrics_per_run.csv'), index=False)
_num = METRICS_PER_RUN.drop(columns=['run']).groupby('condition').agg(['mean', 'std'])
_num.columns = ['{}_{}'.format(a, b) for a, b in _num.columns]
METRICS_SUMMARY = _num.round(3)
METRICS_SUMMARY.to_csv(os.path.join(TAB_DIR, 'metrics_summary_mean_std.csv'))
print(METRICS_SUMMARY.T.to_string())

KEY_METRICS = ['pitch_range', 'pitch_std', 'pitch_entropy_bits', 'pc_entropy_bits', 'onset_density_per_s',
               'ioi_mean_s', 'ioi_cv', 'ioi_entropy_bits', 'dur_unique', 'dur_entropy_bits',
               'ngram_repetition', 'early_density_ratio', 'rest_ratio', 'mean_polyphony',
               'js_pc_to_train', 'js_pitch_to_train']
STAT_TESTS = pd.concat([
    compare_conditions(METRICS_PER_RUN, 'gru_one_note', 'gru_random_uniform', KEY_METRICS).assign(comparison='one_note vs random_uniform'),
    compare_conditions(METRICS_PER_RUN, 'gru_one_note', 'gru_random_unigram', KEY_METRICS).assign(comparison='one_note vs random_unigram'),
], ignore_index=True)
STAT_TESTS.to_csv(os.path.join(TAB_DIR, 'stat_tests_one_note_vs_random.csv'), index=False)
print(STAT_TESTS.round(4).to_string(index=False))


# =============================================================================
# SECTION G -- READABLE FIGURES                           [R1-7][R1-1]
# =============================================================================
def pretty_piano_roll_figure(pm_list, labels, fs, path, t_max=None, dpi=300):
    """[R1-7] Fig. 4 redone: labelled time (s) and pitch (note name + MIDI number) axes,
    octave grid lines, a SHARED pitch/time scale for all panels, and a marginal pitch histogram."""
    rolls = [p.get_piano_roll(fs=fs) for p in pm_list]
    used = [np.where(r.any(axis=1))[0] for r in rolls]
    lo = max(min(int(u.min()) for u in used) - 2, 0)
    hi = min(max(int(u.max()) for u in used) + 2, 127)
    T = max(r.shape[1] / fs for r in rolls)
    if t_max:
        T = min(T, t_max)
    fig, axes = plt.subplots(len(rolls), 2, figsize=(14, 3.8 * len(rolls)), sharey=True,
                             gridspec_kw={'width_ratios': [6, 1]})
    axes = np.atleast_2d(axes)
    c_ticks = [p for p in range(lo, hi + 1) if p % 12 == 0]
    for i, r in enumerate(rolls):
        n_cols = int(T * fs)
        sub = (r[lo:hi + 1, :n_cols] > 0)
        ax = axes[i, 0]
        ax.imshow(sub, origin='lower', aspect='auto', cmap='Oranges', interpolation='nearest', vmin=0, vmax=1,
                  extent=[0, sub.shape[1] / fs, lo - 0.5, hi + 0.5])
        for p in c_ticks:
            ax.axhline(p - 0.5, color='grey', lw=0.5, alpha=0.6)
        ax.set_yticks(c_ticks)
        ax.set_yticklabels(['{} ({})'.format(pretty_midi.note_number_to_name(p), p) for p in c_ticks])
        ax.set_xlim(0, T); ax.set_ylim(lo - 0.5, hi + 0.5)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Pitch (note name, MIDI no.)')
        ax.set_title('({}) {}'.format(chr(97 + i), labels[i]), loc='left', fontweight='bold')
        axh = axes[i, 1]
        axh.barh(np.arange(lo, hi + 1), sub.sum(axis=1) / fs, color='tab:orange', height=0.9)
        axh.set_xlabel('Sounding time (s)'); axh.set_title('Pitch usage')
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.show()


def plot_pitch_class_and_durations(pool, ref_pc_hist, path):
    """[R1-1] Pitch-class distribution and note-duration distribution per condition."""
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    conds = list(pool.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    w = 0.8 / (len(conds) + 1)
    x = np.arange(12)
    axes[0].bar(x - 0.4 + w / 2, ref_pc_hist / ref_pc_hist.sum(), w, label='training data', color='black')
    for k, c in enumerate(conds):
        h = pool[c]['pc']
        axes[0].bar(x - 0.4 + w * (k + 1.5), h / max(h.sum(), 1), w, label=c)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_xlabel('Pitch class'); axes[0].set_ylabel('Relative frequency'); axes[0].set_title('(a) Pitch-class distribution')
    bins = np.linspace(0, 6, 31)
    for c in conds:
        d = np.concatenate(pool[c]['durs']) if pool[c]['durs'] else np.array([])
        if len(d):
            axes[1].hist(np.clip(d, 0, 6), bins=bins, density=True, histtype='step', linewidth=1.6, label=c)
    axes[1].set_xlabel('Note duration (s)'); axes[1].set_ylabel('Density'); axes[1].set_title('(b) Note-duration distribution')
    axes[0].legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.16))   # [R1-7] outside the plot
    axes[1].legend(fontsize=7, loc='upper right')
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches='tight'); plt.show()


pretty_piano_roll_figure([clips['gru_one_note'][0], clips['gru_random_uniform'][0]],
                         ['Single-note start (run 0)', 'Random start (run 0)'], PLAYBACK_FS,
                         os.path.join(FIG_DIR, 'fig4_piano_rolls.png'))
pretty_piano_roll_figure([clips['gru_one_note'][0], clips['gru_random_uniform'][0]],
                         ['Single-note start - first 20 s', 'Random start - first 20 s'], PLAYBACK_FS,
                         os.path.join(FIG_DIR, 'fig4_piano_rolls_first20s.png'), t_max=20)
plot_pitch_class_and_durations(pool, REF_PC_HIST, os.path.join(FIG_DIR, 'pitch_class_and_duration.png'))


# =============================================================================
# SECTION H -- MATERIAL FOR A SMALL LISTENING TEST        [R2-1b]
# Blind, shuffled clips + answer key + empty rating sheet. Rate with >= 5 listeners
# (e.g. musicality 1-5, coherence 1-5, 'human or machine?').
# =============================================================================
def prepare_listening_test(clips, n_per_condition=3, out_dir=LISTEN_DIR, seed_int=SEED):
    rng = random.Random(seed_int)
    items = [(cond, i, pm_obj) for cond, lst in clips.items() for i, pm_obj in enumerate(lst[:n_per_condition])]
    rng.shuffle(items)
    key = []
    for clip_id, (cond, i, pm_obj) in enumerate(items, start=1):
        fname = 'clip_{:02d}.mid'.format(clip_id)
        pm_obj.write(os.path.join(out_dir, fname))
        key.append({'clip_id': clip_id, 'file': fname, 'condition': cond, 'run': i})
    pd.DataFrame(key).to_csv(os.path.join(out_dir, 'answer_key_DO_NOT_SHARE.csv'), index=False)
    pd.DataFrame([{'rater_id': '', 'clip_id': k['clip_id'], 'musicality_1_5': '', 'coherence_1_5': '',
                   'human_or_machine': '', 'comment': ''} for k in key]
                 ).to_csv(os.path.join(out_dir, 'rating_sheet_template.csv'), index=False)
    return pd.DataFrame(key)

LISTENING_KEY = prepare_listening_test(clips)


# =============================================================================
# SECTION I -- ATTENTION VISUALISATION (original, made reusable)
# =============================================================================
def plot_attention(attention, sentence, predicted_sentence, path=None):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()

start_sequence = 150
seq_len = 50

def visualize_attention(model, generate, start_sequence, seq_len):

  test_input = np.array([generate])[:,start_sequence:start_sequence+seq_len]
  out = test_input

  for i, layer in enumerate(model.layers):
    out = layer(out)
    if i == 3 or i == 6:
      print("Attention {}".format(i//3))
      attention = out[1][0].numpy()
      trans_input = test_input[0].astype(str).tolist()
      plot_attention(attention, trans_input, trans_input,
                     path=os.path.join(FIG_DIR, 'attention_layer{}.png'.format(i // 3)))
      out = out[0]

# token list of the first single-note GRU generation (re-created with the same seed as run 0)
_att_tokens = generate_notes(generate_from_one_note(note_tokenizer, ONE_NOTE_START), model, unique_notes,
                             MAX_GENERATE, seq_len, TEMPERATURE, np.random.RandomState(SEED))
visualize_attention(model, _att_tokens, start_sequence, seq_len)
