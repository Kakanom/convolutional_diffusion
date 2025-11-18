from utils.data import get_dataset
from utils.patch_knn import build_patch_database, image_to_patches, knn_search

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


MEAN = 0.5  # для FashionMNIST Normalize(mean=[0.5], std=[0.5])
STD = 0.5


def denormalize_image(img: torch.Tensor) -> torch.Tensor:
    """
    img: [1, H, W] после transforms.ToTensor() + Normalize(0.5, 0.5)

    Вернёт [H, W] в диапазоне [0, 1] для отображения.
    """
    # снимаем нормализацию: x * std + mean
    img = img * STD + MEAN
    img = img.clamp(0, 1)
    return img[0]  # убираем канал


def show_query_and_neighbors(
    dataset,
    kernel_size: int,
    img_idx: int,
    patch_idx: int,
    query_coords,
    neighbours_info,
    savefig: str | None = None,
):
    """
    Рисует:
        - слева: запросное изображение с выделенным патчем
        - справа: top-k соседей (полные изображения с выделенным патчем)

    neighbours_info: список словарей:
        [
          { "img_id": ..., "label": ..., "y": ..., "x": ..., "dist": ... },
          ...
        ]
    """

    k = len(neighbours_info)

    # 1) Картинка-запрос
    query_img, query_label = dataset[img_idx]
    q_img_denorm = denormalize_image(query_img)  # [H, W]
    q_y, q_x = query_coords[patch_idx].tolist()

    # 2) Фигура
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3))

    # ---- СЛЕВА: запрос ----
    ax0 = axes[0]
    ax0.imshow(q_img_denorm.numpy(), cmap="gray")
    ax0.set_title(f"Query img {img_idx}\nlabel={query_label}, patch=({q_y},{q_x})")
    ax0.axis("off")

    rect = Rectangle(
        (q_x, q_y),  # (x, y)
        kernel_size,
        kernel_size,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
    )
    ax0.add_patch(rect)

    # ---- СПРАВА: соседи ----
    for i, info in enumerate(neighbours_info):
        ax = axes[i + 1]

        n_img, n_label = dataset[info["img_id"]]
        n_img_denorm = denormalize_image(n_img)
        y, x = info["y"], info["x"]

        ax.imshow(n_img_denorm.numpy(), cmap="gray")
        ax.set_title(
            f"NN {i+1}\nimg={info['img_id']}, label={n_label}\n"
            f"patch=({y},{x}), d={info['dist']:.3f}"
        )
        ax.axis("off")

        rect_n = Rectangle(
            (x, y),
            kernel_size,
            kernel_size,
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect_n)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight", dpi=200)
        print(f"Figure saved to {savefig}")
    plt.show()


def main():
    # 1. Берём FashionMNIST
    dataset, metadata = get_dataset("fashion_mnist", root="./data")
    print("Dataset:", metadata["name"])
    print("Image size:", metadata["image_size"])
    print("Num channels:", metadata["num_channels"])

    # 2. Строим базу патчей по первым N картинкам
    kernel_size = 7
    max_images = 500  # как у тебя в логе

    print(f"Building patch DB with k={kernel_size}, max_images={max_images}...")
    db_patches, db_meta = build_patch_database(
        dataset,
        kernel_size=kernel_size,
        batch_size=64,
        max_images=max_images,
        device="cpu",
    )
    print("DB patches shape:", db_patches.shape)
    print("DB meta shape:", db_meta.shape)

    # 3. Берём одну картинку как запрос
    img_idx = 0
    image, label = dataset[img_idx]
    print(f"Query image idx={img_idx}, label={label}")

    # 4. Все патчи этой картинки
    query_patches, query_coords = image_to_patches(
        image,
        kernel_size=kernel_size,
        device="cpu",
    )

    patch_idx = 338
    q_patch = query_patches[patch_idx:patch_idx + 1]

    # координаты запросного патча
    q_y, q_x = query_coords[patch_idx].tolist()

    # 5. Ищем k ближайших соседей (берём k+1, чтобы потом выкинуть self-patch)
    k = 5
    dists, idxs = knn_search(
        db_patches,
        q_patch,
        k=k + 1,
        metric="l2",
        device="cpu",
    )

    print(f"\nTop-{k} neighbours for patch #{patch_idx} (coords {(q_y, q_x)}), excluding self-patch:")

    neighbours_info = []
    rank_print = 1

    # идём по всем найденным соседям по порядку,
    # но добавляем только те, которые НЕ совпадают с запросным патчем
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

        neighbours_info.append(
            {
                "img_id": img_id,
                "label": lbl,
                "y": y,
                "x": x,
                "dist": d,
            }
        )

        rank_print += 1
        if len(neighbours_info) == k:
            break

    # 6. Рисуем всё это
    show_query_and_neighbors(
        dataset,
        kernel_size=kernel_size,
        img_idx=img_idx,
        patch_idx=patch_idx,
        query_coords=query_coords,
        neighbours_info=neighbours_info,
        savefig=None, # указать название, чтобы сохранить в папке
    )


if __name__ == "__main__":
    main()
