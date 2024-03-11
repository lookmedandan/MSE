from __future__ import absolute_import, division, print_function

import os
import torch.nn as nn
from pytorch_pretrained_bert.crf import CRF
from pytorch_pretrained_bert.modeling import BertModel

class CWSB(nn.Module):

    def __init__(self, cache_dir, bert_model, num_labels, use_crf=False):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(bert_model, output_attentions=False, cache_dir=cache_dir)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels, bias=False)
        if use_crf:
            self.crf = CRF(tagset_size=self.num_labels-3, gpu=True)
        else:
            self.crf = None

    def forward(self, input_ids, attention_mask=None):
        output, _ = self.bert(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        logits = self.dropout(output)
        logits = self.fc(logits)
        return output, logits

