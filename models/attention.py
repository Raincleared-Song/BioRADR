import torch
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


@MatrixAttention.register("ele_multiply")
class ElementWiseMatrixAttention(MatrixAttention):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.
    """
    def __init__(self) -> None:
        super(ElementWiseMatrixAttention, self).__init__()

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        result = torch.einsum('iaj,ibj->ijab', [tensor_1, tensor_2])
        return result
