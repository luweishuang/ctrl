from __future__ import division
from __future__ import print_function
import tensorflow as tf
import math
import numpy as np
tf.enable_eager_execution()
import argparse
import fastBPE
import platform
from control_codes import CONTROL_CODES

use_py3 = platform.python_version()[0] == '3'

parser = argparse.ArgumentParser(description='TensorFlow code for generating from CTRL')
parser.add_argument('--model_dir', type=str, default='models/seqlen512_v1.ckpt', help='location of model checkpoint')
parser.add_argument('--seed', type=int, default=1337, help='random seed for TensorFlow, numpy and PythonHash')
parser.add_argument('--generate_num', type=int, default=512, help='number of tokens to generate')
parser.add_argument('--temperature', type=float, default=0, help='temperature for sampling distribution; 0 means greedy')
parser.add_argument('--nucleus', type=float, default=0., help='cumulative probability cutoff for nucleus sampling; 0 means no nucleus sampling')
parser.add_argument('--topk', type=int, default=0, help='topk value for sampling from the softmax distribution ; 0 means no topk preferred')
parser.add_argument('--penalty', type=float, default=1.2, help='repetition penalty for greedy sampling')
parser.add_argument('--print_once', action='store_true', help='the completion is printed only at the end; not every word')
parser.add_argument('--topn', type=int, default=0, help='print top-n candidates during generations; defaults to 0 which is no printing')
args = parser.parse_args()






def split_on_window(input_str, over_lap=0.2):
    thr_len = seq_length - 100
    overlap_len = int(over_lap * thr_len)
    output_list = []
    split_sequence = input_str.split()
    iteration_length = len(split_sequence) - (thr_len - overlap_len - 1)
    for index in range(0,  iteration_length, thr_len - overlap_len):
        cur_str = split_sequence[index : index + thr_len]
        output_list.append(cur_str)
    cur_str = split_sequence[index + thr_len - overlap_len:]
    output_list.append(cur_str)
    return output_list


# str_in = "An alternative would be to summarize longer text in chunks. However, this limits the coherence of the final summary as semantic information cannot flow between chunks. On top, finding the right chunking break points is non-trivial, as we have to ensure that at least locally semantic coherent phrases are within the same chunk."
# output_list = split_on_window(str_in)
# print(output_list)


def process_input_text_2_list(input_str, thr_len, overlap=0.2):
    output_list = []
    input_len = len(input_str.split())
    if input_len > thr_len:
        print("input text %d too long, do text summarization by paragraphs! thr_len = %d " % (input_len, thr_len))
        input_list = input_str.split(".")
        para_len = 0
        ii = 0
        while ii < len(input_list):
            cur_len = len(input_list[ii].split(" "))
            para_len += cur_len
            if para_len < thr_len:
                ii += 1
            else:
                para_str = '.'.join(input_list[:ii])
                output_list.append(para_str + '.')
                overlap_sentences = max(int(math.ceil(overlap * ii)), 0)
                del input_list[: ii - overlap_sentences]
                para_len = 0
                ii = 0
        para_str = '.'.join(input_list[:])
        output_list.append(para_str)
    else:
        output_list.append(input_str)
    return output_list


while True:
    # prompt = raw_input('ENTER PROMPT: ') if not use_py3 else input('ENTER PROMPT: ')
    prompt = "An alternative would be to summarize longer text in chunks. However, this limits the coherence of the final summary as semantic information cannot flow between chunks. On top, finding the right chunking break points is non-trivial, as we have to ensure that at least locally semantic coherent phrases are within the same chunk.We extend the standard recurrent Seq2Seq model with pointer-generator to process text across content windows. Attention is performed only at the window-level. A decoder shared across all windows spanning over the respective document poses a link between attentive fragments as the decoder has the ability to preserve semantic information from previous windows. The main idea is to transform the learning objective to a local decision problem. The model slides only in forward direction and processes information from the current window based on the history accumulated over previous windows."
    prompt_list = process_input_text_2_list(prompt, 56)
    print(len(prompt_list))
    summary_list = []
    for ii in range(len(prompt_list)):
        para_str = "News " + prompt_list[ii] + " TL;DR:"
        print(prompt_list[ii])