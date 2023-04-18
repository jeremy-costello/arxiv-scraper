from sentence_transformers import CrossEncoder


def get_reranking(cross_encoder_input, cross_encoder_name):
    cross_encoder = CrossEncoder(f'cross-encoder/{cross_encoder_name}')

    cross_scores = cross_encoder.predict(cross_encoder_input)
    return cross_scores
