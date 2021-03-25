import torch
from torch import Tensor
from torch.nn import Module, EmbeddingBag, Parameter, Softmax
from torchtext.vocab import Vocab


class Ower(Module):

    embedding_bag: EmbeddingBag
    class_embs: Parameter
    multi_weight: Parameter
    multi_bias: Parameter

    def __init__(self, embedding_bag: EmbeddingBag, class_count: int):
        super().__init__()

        self.embedding_bag = embedding_bag

        _, emb_size = embedding_bag.weight.data.shape
        self.class_embs = Parameter(torch.randn(class_count, emb_size))
        self.multi_weight = Parameter(torch.randn(class_count, emb_size))
        self.multi_bias = Parameter(torch.randn(class_count))

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
        :return (batch_size, class_count)
        """

        # Embed token lists
        #
        # < tok_lists_batch  (batch_size, sent_count, sent_len)
        # > sents_batch      (batch_size, sent_count, emb_size)

        sents_batch = self.embed_tok_lists(tok_lists_batch)

        # Calculate attentions (which class matches which sentences)
        #
        # < sents_batch  (batch_size, sent_count, emb_size)
        # > atts_batch   (batch_size, class_count, sent_count)

        atts_batch = self.calc_atts(sents_batch)

        # For each class, mix sentences according to attention
        #
        # < atts_batch   (batch_size, class_count, sent_count)
        # < sents_batch  (batch_size, sent_count, emb_size)
        # > mixes_batch  (batch_size, class_count, emb_size)

        mixes_batch = torch.bmm(atts_batch, sents_batch)

        # Push each mix through its respective single-output linear layer,
        # i.e. scalar multiply each mix vector (of size <emb_size>) with
        # its respective weight vector (of size <emb_size>) and add the
        # bias afterwards.

        logits_batch = torch.einsum('bce, ce -> bc', mixes_batch, self.multi_weight) + self.multi_bias

        return logits_batch

    def embed_tok_lists(self, tok_lists_batch: Tensor) -> Tensor:
        """
        :param tok_lists_batch: (batch_size, sent_count, sent_len)
        :return: (batch_size, sent_count, emb_size)
        """

        # Flatten batch
        #
        # < tok_lists_batch  (batch_size, sent_count, sent_len)
        # > flat_tok_lists   (batch_size * sent_count, sent_len)

        batch_size, sent_count, sent_len = tok_lists_batch.shape

        flat_tok_lists = tok_lists_batch.reshape(batch_size * sent_count, sent_len)

        # Embed token lists
        #
        # < flat_tok_lists  (batch_size * sent_count, sent_len)
        # > flat_sents      (batch_size * sent_count, emb_size)

        flat_sents = self.embedding_bag(flat_tok_lists)

        # Restore batch
        #
        # < flat_sents   (batch_size * sent_count, emb_size)
        # > sents_batch  (batch_size, sent_count, emb_size)

        _, emb_size = flat_sents.shape

        sents_batch = flat_sents.reshape(batch_size, sent_count, emb_size)

        return sents_batch

    def calc_atts(self, sents_batch: Tensor) -> Tensor:
        """
        :param sents_batch: (batch_size, sent_count, emb_size)
        :return: (batch_size, class_count, sent_count)
        """

        # Expand class embeddings for bmm()
        #
        # < self.class_embs   (class_count, emb_size)
        # > class_embs_batch  (batch_size, class_count, emb_size)

        batch_size, _, _ = sents_batch.shape

        class_embs_batch = self.class_embs.unsqueeze(0).expand(batch_size, -1, -1)

        # Multiply each class with each sentence
        #
        # < class_embs_batch  (batch_size, class_count, emb_size)
        # < sents_batch       (batch_size, sent_count, emb_size)
        # > atts_batch        (batch_size, class_count, sent_count)

        atts_batch = torch.bmm(class_embs_batch, sents_batch.transpose(1, 2))

        # Softmax over sentences
        #
        # < atts_batch   (batch_size, class_count, sent_count)
        # > softs_batch  (batch_size, class_count, sent_count)

        softs_batch = Softmax(dim=-1)(atts_batch)

        return softs_batch

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

        # Restore batch shape
        #
        # < flat_sents_batch: FloatTensor[batch_size * sent_count, emb_size]
        # > sents_batch:      FloatTensor[batch_size, sent_count, emb_size]

        _, emb_size = flat_sents_batch.shape

        sents_batch = flat_sents_batch.reshape(batch_size, sent_count, emb_size)

        # Push each sentence through its respective single-output linear layer,
        # i.e. scalar multiply each sentence embeddings with its respective weight
        # vector (of size <emb_size>) and add the bias afterwards.

        logits_batch = torch.einsum('bse, ce -> bsc', sents_batch, self.multi_weight) + self.multi_bias

        return logits_batch

    def bar(self, tok_lists_batch: Tensor) -> Tensor:
        """
        :param tok_lists_batch: (batch_size, sent_count, sent_len)
        :return (batch_size, class_count)
        """

        # Embed token lists
        #
        # < tok_lists_batch  (batch_size, sent_count, sent_len)
        # > sents_batch      (batch_size, sent_count, emb_size)

        sents_batch = self.embed_tok_lists(tok_lists_batch)

        # Calculate attentions (which class matches which sentences)
        #
        # < sents_batch  (batch_size, sent_count, emb_size)
        # > atts_batch   (batch_size, class_count, sent_count)

        atts_batch = self.calc_atts(sents_batch)

        return atts_batch
