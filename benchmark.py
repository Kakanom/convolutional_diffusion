import time

from utils.data import get_dataset
from utils.patch_knn import (
    build_patch_database,
    image_to_patches,
    knn_search,
    build_kdtree,
    knn_search_kdtree,
)


def benchmark_once(max_images, kernel_size=7, k=5, device="cpu"):
    print("\n" + "=" * 80)
    print(f"Benchmark: max_images={max_images}")
    print("=" * 80)

    dataset, metadata = get_dataset("fashion_mnist", root="./data")
    print("Dataset:", metadata["name"])

    # ---------- 1) строим базу патчей ----------
    t0 = time.perf_counter()
    db_patches, db_meta = build_patch_database(
        dataset,
        kernel_size=kernel_size,
        batch_size=64,
        max_images=max_images,
        device=device,
    )
    t_build_db = time.perf_counter() - t0

    print(f"DB patches shape: {db_patches.shape}")
    print(f"DB meta shape:    {db_meta.shape}")
    print(f"Time to build patch DB (max_images={max_images}): {t_build_db:.3f} s")

    # ---------- 2) готовим запросный патч ----------
    img_idx = 0
    image, label = dataset[img_idx]
    print(f"Query image idx={img_idx}, label={label}")

    query_patches, query_coords = image_to_patches(
        image,
        kernel_size=kernel_size,
        device=device,
    )

    Q = query_patches.shape[0]
    center_idx = Q // 2
    q_patch = query_patches[center_idx:center_idx + 1]  # [1, D]
    q_y, q_x = query_coords[center_idx].tolist()
    print(f"Using center patch #{center_idx} with coords=({q_y}, {q_x})")

    # берём k+1, чтобы выкинуть self-patch
    k_with_self = k + 1

    # ---------- 3) brute-force через torch.cdist ----------
    t0 = time.perf_counter()
    bf_dists, bf_idxs = knn_search(
        db_patches,
        q_patch,
        k=k_with_self,
        metric="l2",
        device=device,
    )
    t_bf = time.perf_counter() - t0
    print(f"Brute-force knn_search: {t_bf:.6f} s")

    # ---------- 4) KD-tree: build ----------
    t0 = time.perf_counter()
    tree = build_kdtree(db_patches, metric="l2", leaf_size=40)
    t_tree_build = time.perf_counter() - t0
    print(f"KDTree build time: {t_tree_build:.6f} s")

    # ---------- 5) KD-tree: query ----------
    t0 = time.perf_counter()
    kd_dists, kd_idxs = knn_search_kdtree(
        tree,
        q_patch,
        k=k_with_self,
    )
    t_tree_query = time.perf_counter() - t0
    print(f"KDTree query time: {t_tree_query:.6f} s")

    # ---------- 6) Выводим top-k для проверки ----------
    print(f"\nTop-{k} neighbours (brute force), excluding self-patch:")
    rank_print = 1
    for d, db_idx in zip(bf_dists[0], bf_idxs[0]):
        db_idx = db_idx.item()
        d = d.item()
        img_id, lbl, y, x = db_meta[db_idx].tolist()
        if img_id == img_idx and y == q_y and x == q_x:
            continue
        print(
            f"  {rank_print}) dist={d:.4f}, img_id={img_id}, label={lbl}, "
            f"patch_coord=({y}, {x})"
        )
        rank_print += 1
        if rank_print > k:
            break

    print(f"\nTop-{k} neighbours (KDTree), excluding self-patch:")
    rank_print = 1
    for d, db_idx in zip(kd_dists[0], kd_idxs[0]):
        db_idx = int(db_idx.item())
        d = float(d.item())
        img_id, lbl, y, x = db_meta[db_idx].tolist()
        if img_id == img_idx and y == q_y and x == q_x:
            continue
        print(
            f"  {rank_print}) dist={d:.4f}, img_id={img_id}, label={lbl}, "
            f"patch_coord=({y}, {x})"
        )
        rank_print += 1
        if rank_print > k:
            break

    print("\nSummary (seconds):")
    print(f"  Build patch DB:     {t_build_db:.3f}")
    print(f"  Brute-force query:  {t_bf:.6f}")
    print(f"  KDTree build:       {t_tree_build:.6f}")
    print(f"  KDTree query:       {t_tree_query:.6f}")


def main():
    # Сначала 500 картинок
    benchmark_once(max_images=500, kernel_size=7, k=5, device="cpu")

    # весь датасет
    # benchmark_once(max_images=None, kernel_size=7, k=5, device="cpu")


if __name__ == "__main__":
    main()
