from utils.data import get_dataset
from utils.patch_knn import build_patch_database, image_to_patches, knn_search


def main():
    dataset, metadata = get_dataset("fashion_mnist", root="./data")
    print("Dataset:", metadata["name"])
    print("Image size:", metadata["image_size"])
    print("Num channels:", metadata["num_channels"])

    kernel_size = 7
    max_images = 500

    print(f"Building patch DB with k={kernel_size}, max_images={max_images}...")
    db_patches, db_meta = build_patch_database(
        dataset,
        kernel_size=kernel_size,
        batch_size=64,
        max_images=max_images,
        device="cpu",
    )
    print("DB patches shape:", db_patches.shape)  # [N_patches, C*k*k]
    print("DB meta shape:", db_meta.shape)        # [N_patches, 4]

    # --- запросное изображение ---
    img_idx = 0
    image, label = dataset[img_idx]
    print(f"Query image idx={img_idx}, label={label}")

    # все патчи этой картинки
    query_patches, query_coords = image_to_patches(
        image,
        kernel_size=kernel_size,
        device="cpu",
    )

    # берём центральный патч
    Q = query_patches.shape[0]
    center_idx = Q // 2
    q_patch = query_patches[center_idx:center_idx + 1]  # [1, D]
    q_y, q_x = query_coords[center_idx].tolist()

    k = 5

    # берём k+1 соседей, чтобы после удаления self-patch осталось k
    dists, idxs = knn_search(
        db_patches,
        q_patch,
        k=k + 1,
        metric="l2",
        device="cpu",
    )

    print(f"\nTop-{k} neighbours for patch #{center_idx} (coords {(q_y, q_x)}), excluding self-patch:")

    rank_print = 1
    for d, db_idx in zip(dists[0], idxs[0]):
        db_idx = db_idx.item()
        d = d.item()
        img_id, lbl, y, x = db_meta[db_idx].tolist()

        # выкидываем ровно тот же патч (та же картинка, те же координаты)
        if img_id == img_idx and y == q_y and x == q_x:
            continue

        print(
            f"  {rank_print}) dist={d:.4f}, img_id={img_id}, label={lbl}, "
            f"patch_coord=({y}, {x})"
        )
        rank_print += 1

        if rank_print > k:
            break


if __name__ == "__main__":
    main()
