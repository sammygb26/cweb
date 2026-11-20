from cweb.utils import *
from cweb.helpers import random_phases, trees_all_close

import jax
import jax.numpy as jnp

import os

from test_helpers.filesystem import folder

def test_data_logger_creates_folder(folder):
    assert not os.path.exists(folder)
    Logger(folder)
    assert os.path.exists(folder)

def test_data_logger_works_with_premade_folder(folder):
    os.makedirs(folder)
    Logger(folder)

def test_data_logger_saves_jax_tensor(folder):
    l = Logger(folder)

    lines = ['ab', 'ba']
    for line in lines:
        l.log(line)

    with open(l.path, 'r') as file:
        read_lines = file.readlines()

        for line, read_line in zip(lines, read_lines):
            assert line + '\n' ==  read_line

    

