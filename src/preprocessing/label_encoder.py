"""
Label encoding utilities for NIH Chest X-ray dataset
"""

# List of diseases used in the project

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]


def encode_labels(label_string):
    """
    Convert label string into multi-label vector.

    Example:
    'Atelectasis|Effusion' → [1,0,1,0,0...]
    """

    labels = label_string.split("|")

    encoded = [0] * len(DISEASES)

    for i, disease in enumerate(DISEASES):
        if disease in labels:
            encoded[i] = 1

    return encoded