# I've chosen small variances because I think they can't exceed 1, true?
# Let's say that 0.5 <= mean <= 0.01
# I put always the same noise_variance (0.005)
configuration = [
    # Subcampaign 1 (Amazon subcampaign)(women)
    # the 1st () indicates the type of website and its consequence
    # is the 2nd () which is the class containing 2 classes
    [  # CLASS 1
        {  # INTERESTED USERS
            'class_name': 'Young men (1)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.5,
                    'variance': 0.2
                },
                #     SLOT 2
                {
                    'mean': 0.4,
                    'variance': 0.05
                },
                #     SLOT 3
                {
                    'mean': 0.2,
                    'variance': 0.05
                },
                #     SLOT 4
                {
                    'mean': 0.05,
                    'variance': 0.05
                }
            ]
        },
        # CLASS 2
        {  # INTERESTED USERS
            'class_name': 'Old men (3)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.3,
                    'variance': 0.25
                },
                #     SLOT 2
                {
                    'mean': 0.12,
                    'variance': 0.15
                },
                #     SLOT 3
                {
                    'mean': 0.05,
                    'variance': 0.1
                },
                #     SLOT 4
                {
                    'mean': 0.04,
                    'variance': 0.1
                }
            ]
        },
        {  # CLASS 3
            # INTERESTED USERS
            'class_name': 'Women (0&2)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.18,
                    'variance': 0.5
                },
                #     SLOT 2
                {
                    'mean': 0.08,
                    'variance': 0.45
                },
                #     SLOT 3
                {
                    'mean': 0.05,
                    'variance': 0.35
                },
                #     SLOT 4
                {
                    'mean': 0.04,
                    'variance': 0.35
                }
            ]
        }

    ],

    # Subcampaign 2 (Rollex)(women)
    # higher means
    [  # CLASS 1
        {  # INTERESTED USERS
            'class_name': 'Young men (1)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.5,
                    'variance': 0.2
                },
                #     SLOT 2
                {
                    'mean': 0.45,
                    'variance': 0.1
                },
                #     SLOT 3
                {
                    'mean': 0.35,
                    'variance': 0.1
                },
                #     SLOT 4
                {
                    'mean': 0.3,
                    'variance': 0.1
                }
            ]
        },
        # CLASS 2
        {  # INTERESTED USERS
            'class_name': 'Old men (3)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.38,
                    'variance': 0.2
                },
                #     SLOT 2
                {
                    'mean': 0.26,
                    'variance': 0.15
                },
                #     SLOT 3
                {
                    'mean': 0.19,
                    'variance': 0.1
                },
                #     SLOT 4
                {
                    'mean': 0.15,
                    'variance': 0.1
                }
            ]
        },
        {  # CLASS 3
            # INTERESTED USERS
            'class_name': 'Women (0&2)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.35,
                    'variance': 0.45
                },
                #     SLOT 2
                {
                    'mean': 0.3,
                    'variance': 0.4
                },
                #     SLOT 3
                {
                    'mean': 0.27,
                    'variance': 0.3
                },
                #     SLOT 4
                {
                    'mean': 0.23,
                    'variance': 0.3
                }
            ]
        }

    ],
    # Subcampaign 3 (Presents website)(old)
    [
        # CLASS 1
        {  # INTERESTED USERS
            'class_name': 'Young women (0)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.3,
                    'variance': 0.3
                },
                #     SLOT 2
                {
                    'mean': 0.17,
                    'variance': 0.2
                },
                #     SLOT 3
                {
                    'mean': 0.05,
                    'variance': 0.2
                },
                #     SLOT 4
                {
                    'mean': 0.01,
                    'variance': 0.2
                }
            ]
        },
        # CLASS 2
        {  # INTERESTED USERS
            'class_name': 'Young men (1)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.3,
                    'variance': 0.2
                },
                #     SLOT 2
                {
                    'mean': 0.24,
                    'variance': 0.15
                },
                #     SLOT 3
                {
                    'mean': 0.19,
                    'variance': 0.15
                },
                #     SLOT 4
                {
                    'mean': 0.15,
                    'variance': 0.15
                }
            ]
        },
        # CLASS 3
        {  # INTERESTED USERS
            'class_name': 'Old (2&3)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.5,
                    'variance': 0.2
                },
                #     SLOT 2
                {
                    'mean': 0.3,
                    'variance': 0.1
                },
                #     SLOT 3
                {
                    'mean': 0.1,
                    'variance': 0.1
                },
                #     SLOT 4
                {
                    'mean': 0.05,
                    'variance': 0.1
                }
            ]
        }
    ],
    # Subcampaign 4 (women discount website)(men)
    [
        # CLASS 1
        {
            'class_name': 'Young women (0)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.2,
                    'variance': 0.2
                },
                #     SLOT 2
                {
                    'mean': 0.1,
                    'variance': 0.1
                },
                #     SLOT 3
                {
                    'mean': 0.05,
                    'variance': 0.1
                },
                #     SLOT 4
                {
                    'mean': 0.02,
                    'variance': 0.1
                }
            ]
        },
        # CLASS 2
        {
            'class_name': 'Old women (2)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.25,
                    'variance': 0.3
                },
                #     SLOT 2
                {
                    'mean': 0.08,
                    'variance': 0.25
                },
                #     SLOT 3
                {
                    'mean': 0.03,
                    'variance': 0.2
                },
                #     SLOT 4
                {
                    'mean': 0.01,
                    'variance': 0.2
                }
            ]
        },
        # CLASS 3
        {
            'class_name': 'Men (1&3)',
            'noise_mean': 0,
            'noise_variance': 0.005,
            'max_value': 80,
            'offset': 0,
            'speed': 0.05,
            'click_probabilities': [
                #     SLOT 1
                {
                    'mean': 0.4,
                    'variance': 0.05
                },
                #     SLOT 2
                {
                    'mean': 0.3,
                    'variance': 0.05
                },
                #     SLOT 3
                {
                    'mean': 0.2,
                    'variance': 0.05
                },
                #     SLOT 4
                {
                    'mean': 0.1,
                    'variance': 0.05
                }
            ]
        }
    ]
]


def get_configuration():
    return configuration
