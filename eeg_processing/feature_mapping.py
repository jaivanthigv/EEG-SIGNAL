import numpy as np

def map_to_text(features):
    """
    Map features to predefined thoughts/sentences (mock classifier).
    """
    sentences = [
        "I want to eat something.",
        "I feel happy today.",
        "I'm thinking about work.",
        "It's a beautiful day outside.",
        "I need some rest."
    ]
    
    feature_sum = np.sum(features)
    if feature_sum < 10:
        return sentences[0]
    elif feature_sum < 20:
        return sentences[1]
    elif feature_sum < 30:
        return sentences[2]
    elif feature_sum < 40:
        return sentences[3]
    else:
        return sentences[4]
