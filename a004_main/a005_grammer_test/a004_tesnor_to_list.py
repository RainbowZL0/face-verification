import torch
from torch.nn import functional as F


def test_01():
    a = torch.rand(size=(5, 6), dtype=torch.float32)
    b = torch.rand(size=(5, 6), dtype=torch.float32)
    cos = F.cosine_similarity(x1=a, x2=b, dim=1)
    pass


def test_02():
    def test_unpack(a, b):
        print(a, b)

    dic = {"a": 1, "b": 2}
    test_unpack(**dic)
    print(*dic)


if __name__ == "__main__":
    pass
    test_02()
