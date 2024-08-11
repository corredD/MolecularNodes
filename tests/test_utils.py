import molecularnodes as mn
import bpy
import numpy as np

mn._test_register()


def test_correct_1d():
    assert np.allclose(
        mn.utils.correct_periodic_1d(np.array((0.9, 0.1)), np.array((0.1, 0.9)), 1.0),
        np.array((1.1, -0.1)),
    )