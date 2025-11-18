import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.neighbors import KDTree


@torch.no_grad()
def build_patch_database(
    dataset,
    kernel_size: int = 7,
    batch_size: int = 64,
    max_images: int | None = None,
    device: str | torch.device = "cpu",
):
    """
    Строит базу патчей из датасета.

    Возвращает:
        patches: Tensor [N_patches, C * k * k]  -- векторы патчей
        meta:    Tensor [N_patches, 4]          -- (img_idx, label, y, x)
                 img_idx  – индекс картинки в датасете
                 label    – метка класса (для cifar10/mnist)
                 y, x     – координата верхнего левого угла патча
    """

    device = torch.device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    patches_list = []
    meta_list = []

    img_counter = 0  # глобальный индекс картинки в датасете

    for images, labels in loader:
        b, c, h, w = images.shape

        # ограничиваем максимальное число картинок
        if max_images is not None and img_counter >= max_images:
            break

        if max_images is not None and img_counter + b > max_images:
            # убираем лишние
            needed = max_images - img_counter
            images = images[:needed]
            labels = labels[:needed]
            b = needed

        images = images.to(device)
        labels = labels.to(device)

        # F.unfold: [B, C, H, W] -> [B, C*k*k, L],
        # L = (H-k+1)*(W-k+1)
        unfolded = F.unfold(
            images, kernel_size=kernel_size, stride=1, padding=0
        )  # [B, C*k*k, L]

        # -> [B, L, C*k*k] -> [B*L, C*k*k]
        unfolded = unfolded.permute(0, 2, 1).reshape(-1, c * kernel_size * kernel_size)
        patches_list.append(unfolded.cpu())

        # координаты патчей внутри картинки
        hp = h - kernel_size + 1
        wp = w - kernel_size + 1
        ys = torch.arange(hp)
        xs = torch.arange(wp)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [hp, wp]
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)  # [L, 2]
        coords = coords.to(torch.long)

        # размножаем coords и метки по всем картинкам в батче
        coords = coords.unsqueeze(0).repeat(b, 1, 1)  # [B, L, 2] -> [B*L, 2]
        coords = coords.reshape(-1, 2)

        img_indices = torch.arange(img_counter, img_counter + b).to(torch.long)
        img_indices = img_indices.unsqueeze(1).repeat(1, hp * wp).reshape(-1, 1)

        labels_rep = labels.view(-1, 1).repeat(1, hp * wp).reshape(-1, 1)

        meta = torch.cat([img_indices, labels_rep.cpu(), coords.cpu()], dim=1)  # [B*L, 4]
        meta_list.append(meta)

        img_counter += b

    patches = torch.cat(patches_list, dim=0) if patches_list else torch.empty(0)
    meta = torch.cat(meta_list, dim=0) if meta_list else torch.empty(0, 4, dtype=torch.long)

    return patches, meta


@torch.no_grad()
def image_to_patches(
    image: torch.Tensor,
    kernel_size: int = 7,
    device: str | torch.device = "cpu",
):
    """
    Вырезает все k*k патчи из одной картинки.

    Аргументы:
        image: Tensor [C, H, W]
    Возвращает:
        patches: Tensor [L, C*k*k]
        coords:  Tensor [L, 2]  -- (y, x) верхнего левого угла патча
    """
    device = torch.device(device)
    c, h, w = image.shape

    img = image.unsqueeze(0).to(device)  # [1, C, H, W]

    unfolded = F.unfold(img, kernel_size=kernel_size, stride=1, padding=0)  # [1, C*k*k, L]
    unfolded = unfolded.permute(0, 2, 1).reshape(-1, c * kernel_size * kernel_size)  # [L, C*k*k]

    hp = h - kernel_size + 1
    wp = w - kernel_size + 1
    ys = torch.arange(hp)
    xs = torch.arange(wp)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).to(torch.long)  # [L, 2]

    return unfolded.cpu(), coords.cpu()


@torch.no_grad()
def knn_search(
    database_patches: torch.Tensor,
    query_patches: torch.Tensor,
    k: int = 10,
    metric: str = "l2",
    device: str | torch.device = "cpu",
):
    """
    Находит k ближайших соседей для query_patches в базе database_patches.

    Аргументы:
        database_patches: [N, D]
        query_patches:    [Q, D]
        k:                число соседей
        metric:           'l2' или 'l1'

    Возвращает:
        dists: Tensor [Q, k]  -- расстояния до соседей
        idx:   Tensor [Q, k]  -- индексы соседей в database_patches
    """
    device = torch.device(device)

    db = database_patches.to(device)   # [N, D]
    q = query_patches.to(device)       # [Q, D]

    if metric == "l2":
        dists = torch.cdist(q, db, p=2)   # [Q, N]
    elif metric == "l1":
        dists = torch.cdist(q, db, p=1)   # [Q, N]
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # берём k минимальных расстояний
    dists_k, idx_k = torch.topk(dists, k, largest=False, dim=1)

    return dists_k.cpu(), idx_k.cpu()


# ==========================
# KD-TREE ВЕРСИЯ
# ==========================

def build_kdtree(
    database_patches: torch.Tensor,
    metric: str = "l2",
    leaf_size: int = 40,
):
    """
    Строит KD-дерево по базе патчей.

    Аргументы:
        database_patches: Tensor [N, D] (на CPU или GPU — всё равно, сконвертим в numpy)
        metric:           'l2' или 'l1'
        leaf_size:        параметр KDTree (баланс точность/скорость)

    Возвращает:
        tree: sklearn.neighbors.KDTree
    """
    if metric == "l2":
        sk_metric = "euclidean"
    elif metric == "l1":
        sk_metric = "manhattan"
    else:
        raise ValueError(f"Unknown metric for KDTree: {metric}")

    # KDTree работает с numpy, поэтому переносим на CPU и в numpy
    data_np = database_patches.detach().cpu().numpy()
    tree = KDTree(data_np, leaf_size=leaf_size, metric=sk_metric)
    return tree


@torch.no_grad()
def knn_search_kdtree(
    tree: KDTree,
    query_patches: torch.Tensor,
    k: int = 10,
):
    """
    KNN-поиск через заранее построенное KD-дерево.

    Аргументы:
        tree:          KDTree, построенный по database_patches
        query_patches: Tensor [Q, D]
        k:             число соседей

    Возвращает:
        dists: Tensor [Q, k]  -- расстояния до соседей (numpy -> torch)
        idx:   Tensor [Q, k]  -- индексы соседей в исходном database_patches
    """
    q_np = query_patches.detach().cpu().numpy()  # [Q, D]
    dists_np, idx_np = tree.query(q_np, k=k)     # [Q, k], [Q, k]

    dists = torch.from_numpy(dists_np)
    idx = torch.from_numpy(idx_np).long()

    return dists, idx
