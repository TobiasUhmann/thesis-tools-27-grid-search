import torch
from torch import Tensor
from torch.nn import Module, EmbeddingBag, Linear, Parameter
from torchtext.vocab import Vocab


class BaseConcat(Module):

    embedding_bag: EmbeddingBag
    linear: Linear

    def __init__(self, embedding_bag: EmbeddingBag, class_count: int):
        super().__init__()

        self.embedding_bag = embedding_bag

        _, emb_size = embedding_bag.weight.data.shape
        self.linear = Linear(emb_size, class_count)

    @classmethod
    def from_random(cls, vocab_size: int, emb_size: int, class_count: int, mode: str):
        embedding_bag = EmbeddingBag(num_embeddings=vocab_size, embedding_dim=emb_size, mode=mode)

        return cls(embedding_bag, class_count)

    @classmethod
    def from_pre_trained(cls, vocab: Vocab, class_count: int, mode: str, update_vectors: bool):
        embedding_bag = EmbeddingBag.from_pretrained(vocab.vectors, mode=mode, freeze=(not update_vectors))

        return cls(embedding_bag, class_count)

    def forward(self, tok_lists_batch: Tensor) -> Tensor:
        """
        :param tok_lists_batch: (batch_size, sent_count, sent_len)
        :return: (batch_size, class_count)
        """

        # Concat entity's sentences to a single context
        #
        # < tok_lists_batch  (batch_size, sent_count, sent_len)
        # > tok_list_batch   (batch_size, sent_count * sent_len)

        batch_size, sent_count, sent_len = tok_lists_batch.shape

        tok_list_batch = tok_lists_batch.reshape(batch_size, sent_count * sent_len)

        # Embed context
        #
        # < tok_list_batch  (batch_size, sent_count * sent_len)
        # > ctxt_batch      (batch_size, emb_size)

        ctxt_batch = self.embedding_bag(tok_list_batch)

        # # Normalize context before linear layer
        # #
        # # < ctxt_batch  (batch_size, emb_size)
        # # > ctxt_batch  (batch_size, emb_size)
        #
        # ctxt_batch = (ctxt_batch - ctxt_batch.min()) / (ctxt_batch.max() - ctxt_batch.min())

        # Push context through linear layer
        #
        # < ctxt_batch    (batch_size, emb_size)
        # > logits_batch  (batch_size, class_count)

        logits_batch = self.linear(ctxt_batch)

        return logits_batch

    def foo(self, tok_lists_batch: Tensor) -> Tensor:
        """
        :param tok_lists_batch: IntTensor[batch_size, sent_count, sent_len]

        :return: logits_batch: FloatTensor[batch_size, class_count]
        """

        # Flatten sentences
        #
        # < tok_lists_batch:      IntTensor[batch_size, sent_count, sent_len]
        # > flat_tok_lists_batch: IntTensor[batch_size * sent_count, sent_len]

        batch_size, sent_count, sent_len = tok_lists_batch.shape

        flat_tok_lists_batch = tok_lists_batch.reshape(batch_size * sent_count, sent_len)

        # Embed sentences
        #
        # < flat_tok_lists_batch: IntTensor[batch_size * sent_count, sent_len]
        # > flat_sents_batch:     FloatTensor[batch_size * sent_count, emb_size]

        flat_sents_batch = self.embedding_bag(flat_tok_lists_batch)

        # Push sentences through linear layer
        #
        # < flat_sents_batch:  FloatTensor[batch_size * sent_count, emb_size]
        # > flat_logits_batch: FloatTensor[batch_size * sent_count, class_count]

        flat_logits_batch = self.linear(flat_sents_batch)

        # Restore batch shape
        #
        # < flat_logits_batch: FloatTensor[batch_size * sent_count, class_count]
        # > logits_batch:      FloatTensor[batch_size, sent_count, class_count]

        _, class_count = flat_logits_batch.shape

        logits_batch = flat_logits_batch.reshape(batch_size, sent_count, class_count)

        return logits_batch
