import logging
import os

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import DebertaV2Model, DebertaV2PreTrainedModel

from utils import MODEL_PATH, get_intent_labels, get_slot_labels, get_latest_ckpt
from model.module import IntentClassifier, SlotClassifier

logger = logging.getLogger(__name__)

class JointDeBERTa(DebertaV2PreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointDeBERTa, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.deberta = DebertaV2Model.from_pretrained(MODEL_PATH)

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    @classmethod
    def load_model(cls, model_dir, args, device, intent_labels=None, slot_labels=None):
        # Check whether model exists
        if not os.path.exists(model_dir):
            raise Exception("Model doesn't exist! Train first!")

        if not hasattr(args, "checkpoint") or args.checkpoint == -1:
            model_file = get_latest_ckpt(model_dir)
        else:
            model_file = f"{model_dir}/checkpoint-{args.checkpoint}"

        logger.info(f"Loading model from {model_file}")
        try:
            if intent_labels is None:
                intent_labels = get_intent_labels(args)
            if slot_labels is None:
                slot_labels = get_slot_labels(args)
            model = JointDeBERTa.from_pretrained(model_file,
                                                 args=args,
                                                 intent_label_lst=intent_labels,
                                                 slot_label_lst=slot_labels)
            model.to(device)
            logger.info("***** Model Loaded *****")
        except:
            try:
                # mainly for backwards compatibility
                model = torch.load(model_file, map_location=device)
            except:
                raise Exception("Some model files might be missing...")
        model.eval()
        return model

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.deberta(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
