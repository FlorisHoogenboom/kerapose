ELEMENTS = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb",
            "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank",
            "Leye", "Reye", "Lear", "Rear", "pt19"]

# Map id to element name
ELEMENTS_MAP = dict(enumerate(ELEMENTS))

# Confidence threshold for determination whether maximum on heat map constitutes a joint
JOINT_CONFIDENCE_THRESHOLD = .1

NETWORK_N_STAGES = 6
NETWORK_N_OUTPUT_HM_BRANCH = 19
NETWORK_N_OUTPUT_PAF_BRANCH = 38

PREDICT_STACK_SIZES = [184, 368, 552, 736]
LINKS = [(1, 8), (11, 12), (8, 9), (9, 10), (12, 13), (1, 0),
         (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (14, 16),
         (0, 15), (2, 16), (15, 17), (5, 17), (1, 11), (0, 14)]
