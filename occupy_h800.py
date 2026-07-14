import torch
import torch.multiprocessing as mp


def run(rank):
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    print(f"GPU {rank} start")

    # 足够大的矩阵，打 Tensor Core
    n = 32768

    a = torch.randn(
        n, n,
        device=device,
        dtype=torch.float16
    )

    b = torch.randn(
        n, n,
        device=device,
        dtype=torch.float16
    )

    torch.cuda.synchronize()

    while True:
        c = torch.matmul(a, b)

        # 防止计算被优化掉
        torch.cuda.synchronize()


if __name__ == "__main__":
    mp.spawn(
        run,
        nprocs=8
        )
