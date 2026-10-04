"""Split paired daily series into training years and one held-out year."""

import numpy as np


def split_leave_one_year_out(input_data, output_data, year_index, days_per_year=112):
    inputs = np.asarray(input_data)
    outputs = np.asarray(output_data)
    if inputs.ndim != 1 or outputs.ndim != 1 or inputs.shape != outputs.shape:
        raise ValueError("CIO and precipitation must be aligned one-dimensional series")
    if not isinstance(days_per_year, (int, np.integer)) or days_per_year <= 0:
        raise ValueError("days_per_year must be a positive integer")
    if inputs.size % days_per_year or inputs.size < 2 * days_per_year:
        raise ValueError("At least two complete year blocks are required")
    year_count = inputs.size // days_per_year
    if not isinstance(year_index, (int, np.integer)) or not 0 <= year_index < year_count:
        raise ValueError("Target year index is outside the available year blocks")

    start = days_per_year * year_index
    end = start + days_per_year
    train_inputs = np.concatenate((inputs[:start], inputs[end:]))
    train_outputs = np.concatenate((outputs[:start], outputs[end:]))
    return train_inputs, train_outputs, inputs[start:end], outputs[start:end]
