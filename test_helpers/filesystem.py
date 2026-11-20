import pytest
import random
import shutil
import os

UNIT_TEST_ACTIVE_FOLDER = 'unit_test_active_folder'

@pytest.fixture
def folder():
    folder = os.path.join(
        UNIT_TEST_ACTIVE_FOLDER,
        f'{random.randint(0, 0xFFFF_FFFF):08X}_test_folder'
    )

    yield folder

    shutil.rmtree(UNIT_TEST_ACTIVE_FOLDER)