import logging
import torch
import transformers
# hide the loading messages
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
from bert_score import score

ref = 'The young man in a slicker.'
hyp1 = 'The boy in a coat.'
hyp2 = 'The man in a coat.'

print('ot score1: ', score([hyp1], [ref], epsilon=0.009, reg1=0.23, reg2=0.31, lang='en', idf=False, lemd=True))
