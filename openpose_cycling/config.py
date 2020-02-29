# We use None to indicate the layers of the heatmap.
ELEMENTS = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb",
            "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank",
            "Leye", "Reye", "Lear", "Rear", None]

# Map id to element name
ELEMENTS_MAP = dict(enumerate(ELEMENTS))

# The links correspond to the elements and indicate which joints should be connected
# The PAF fields are also mapped to these tuples and indicate the affinity of
# two joint locations in the picture.
LINKS = [(1, 8),  (8, 9),

         (9, 10), (1, 11),

         (11, 12), (12, 13),

         (1, 1),  # What is this one???
         (2, 3),

         (3, 4),
         (17, 2),    # Does this one make sense?

         (1, 5),
         (5, 6),



         (6, 7),
         (1, 5),   # This is unkown


         (0, 1),
         (0, 15),

         (0, 14),
         (15, 17),

        (2, 16) ]


{0: 'nose',
 1: 'neck',
 2: 'Rsho',
 3: 'Relb',
 4: 'Rwri',
 5: 'Lsho',
 6: 'Lelb',
 7: 'Lwri',
 8: 'Rhip',
 9: 'Rkne',
 10: 'Rank',
 11: 'Lhip',
 12: 'Lkne',
 13: 'Lank',
 14: 'Leye',
 15: 'Reye',
 16: 'Lear',
 17: 'Rear',
 18: None}

# Confidence threshold for determination whether maximum on heat map constitutes a joint
JOINT_CONFIDENCE_THRESHOLD = .1

NETWORK_N_STAGES = 6
NETWORK_N_OUTPUT_HM_BRANCH = len(ELEMENTS)
NETWORK_N_OUTPUT_PAF_BRANCH = len(LINKS) * 2

PREDICT_STACK_SIZES = [(200, 184), (392, 368), (584, 552)]
POSTPROCESS_INPUT_SIZE = 512

FFMPEG_PIX_FMT = 'yuv420p'

# TODO: Scales multiplier effect
# TODO: PAF averaging
# TODO: Links mapping
